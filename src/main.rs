use burn::record::Recorder;
use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    record::CompactRecorder,
    tensor::activation::softmax,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            AccuracyMetric, CpuMemory, CpuUse, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};
use derive_new::new;
use rand::Rng;
use rayon::iter::ParallelIterator;
use std::env;
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::Read,
    u8,
};

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    optim::AdamWConfig,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Data, Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use rayon::iter::IntoParallelRefIterator;

pub struct TextDataset {
    block_size: usize,
    content: Vec<i16>,
}

fn encode(data: &str, stoi: &BTreeMap<char, i16>) -> Vec<i16> {
    let mut output = Vec::with_capacity(data.len());
    for c in data.chars() {
        if let Some(v) = stoi.get(&c) {
            output.push(*v);
        }
    }
    output
}

fn decode(data: &[i16], itos: &BTreeMap<i16, char>) -> String {
    let mut output = Vec::with_capacity(data.len());
    for i in data {
        if let Some(v) = itos.get(i) {
            output.push(*v as u8);
        }
    }
    String::from_utf8(output).unwrap()
}

fn get_mappings(text: &str) -> (BTreeMap<char, i16>, BTreeMap<i16, char>, Vec<char>) {
    //let vocab: HashSet<char> = HashSet::from_iter(text.chars());
    //let mut vocab = vocab.iter().copied().collect::<Vec<char>>();
    let mut vocab = vec![
        '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', '\\', '[', ']', '{', '}', ' ',
        '|', ':', ';', '"', '\'', '<', '>', ',', '.', '?', '/', '~', '`', 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x',
        'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z',
    ];
    vocab.sort();
    println!("vocab: {:#?}", vocab);
    println!("vocab length: {}", vocab.len());
    let mut stoi = BTreeMap::new();
    let mut itos = BTreeMap::new();
    for (i, c) in vocab.iter().enumerate() {
        itos.insert(i as i16, *c);
        stoi.insert(*c, i as i16);
    }
    (stoi, itos, vocab)
}

impl TextDataset {
    pub fn new(dataset: &str, split: bool) -> Self {
        // todo replace with buffered
        let mut text_data = File::open(dataset).unwrap();
        let mut data_contents = String::new();
        text_data
            .read_to_string(&mut data_contents)
            .unwrap_or_else(|_| panic!("read contents from {} should success", dataset));

        let (stoi, _, _) = get_mappings(&data_contents);

        let text_slice = data_contents.as_str();
        let text_len = text_slice.len();
        let split_idx = text_len * 7 / 10;
        let text_slice = if split {
            &text_slice[0..split_idx]
        } else {
            &text_slice[split_idx + 1..text_len]
        };

        let content = encode(text_slice, &stoi)
            .par_iter()
            .map(|item| *item)
            .collect::<Vec<i16>>();

        Self {
            block_size: 8,
            content,
        }
    }
}

impl Dataset<ContextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<ContextItem> {
        if index > self.content.len() - self.block_size - 1 {
            return None;
        }
        let context = self.content.as_slice()[index..index + self.block_size].to_vec();
        let target = self.content.as_slice()[index + 1..index + self.block_size + 1].to_vec();
        Some(ContextItem { context, target })
    }

    fn len(&self) -> usize {
        self.content.len() - self.block_size
    }
}

#[derive(Debug, Clone)]
pub struct ContextItem {
    context: Vec<i16>,
    target: Vec<i16>,
}

#[derive(new)]
pub struct BigramBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Debug, Clone)]
pub struct BigramBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<ContextItem, BigramBatch<B>> for BigramBatcher<B> {
    fn batch(&self, items: Vec<ContextItem>) -> BigramBatch<B> {
        let inputs = items
            .par_iter()
            .map(|item| &item.context)
            .map(|context| Data::from(context.as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device))
            .collect::<Vec<Tensor<B, 1, Int>>>();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .par_iter()
            .map(|item| &item.target)
            .map(|context| Data::from(context.as_slice()))
            .map(|data| Tensor::from_data(data.convert(), &self.device))
            .collect::<Vec<Tensor<B, 1, Int>>>();
        let targets = Tensor::stack(targets, 0);

        BigramBatch { inputs, targets }
    }
}

#[derive(Module, Debug)]
pub struct BigramLanageModel<B: Backend> {
    embedding: Embedding<B>,
    linear: Linear<B>,
}

impl<B: Backend> BigramLanageModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        self.linear.forward(x)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(input);
        let [b, t, c] = output.dims();
        let [b_prim, t_prim] = targets.dims();
        let output = output.reshape([b * t, c]);
        let targets = targets.reshape([b_prim * t_prim]);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<BigramBatch<B>, ClassificationOutput<B>>
    for BigramLanageModel<B>
{
    fn step(&self, batch: BigramBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BigramBatch<B>, ClassificationOutput<B>> for BigramLanageModel<B> {
    fn step(&self, batch: BigramBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

#[derive(Config)]
pub struct BigramLanageModelConfig {
    #[config(default = 65)]
    vocab_size: usize,
    #[config(default = 130)]
    d_model: usize,
}

impl BigramLanageModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramLanageModel<B> {
        BigramLanageModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        record: BigramLanageModelRecord<B>,
    ) -> BigramLanageModel<B> {
        BigramLanageModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model)
                .init_with(record.embedding),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init_with(record.linear),
        }
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: BigramLanageModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 4)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learing_rate: f64,
}

pub fn generate<B: Backend>(
    static_dir: &str,
    device: B::Device,
    input: &str,
    max_new_token: usize,
    stoi: &BTreeMap<char, i16>,
    itos: &BTreeMap<i16, char>,
) {
    let config = TrainingConfig::load(format!("{static_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{static_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let encoded_str = encode(input, stoi);
    let context = encoded_str.as_slice();
    let mut input = Tensor::<B, 1, Int>::from_data(Data::from(context).convert(), &device);

    for _ in 0..max_new_token {
        // println!("input: {:?}", input.to_data().value);
        let [input_dim] = input.dims();
        let logits = model.forward(input.clone().reshape([1, input_dim]));
        let [b, t, c] = logits.dims();
        let probs: Tensor<B, 2> = softmax(logits.slice::<3>([0..b, t - 1..t, 0..c]).squeeze(1), 1);
        let prob_elems = probs.to_data().convert::<f32>().value;
        // println!("prob_elems: {:?}", prob_elems.len());
        let elem_next = sample_distribution(&prob_elems);
        // println!("elem_next={:?}", elem_next);
        let input_next =
            Tensor::<B, 1, Int>::from(Data::from([elem_next as i16]).convert()).to_device(&device);
        input = Tensor::cat(vec![input.clone(), input_next], 0);
    }

    println!(
        "{:?}",
        decode(input.to_data().convert::<i16>().value.as_slice(), itos)
    );
}

fn sample_distribution(distribution: &[f32]) -> usize {
    let mut cdf = Vec::with_capacity(distribution.len());
    let mut sum = 0.0;
    for &prob in distribution.iter() {
        sum += prob;
        cdf.push(sum);
    }

    // Normalize the CDF if necessary
    let cdf_last = *cdf.last().unwrap();
    if cdf_last != 1.0 {
        for cdf_val in cdf.iter_mut() {
            *cdf_val /= cdf_last;
        }
    }

    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(0f32..1f32);

    // Step 4: Find the index in the CDF
    cdf.iter()
        .position(|&x| x >= random_value)
        .unwrap_or_else(|| cdf.len() - 1)
}

type MyBackend = LibTorch;
type B = Autodiff<MyBackend>;

fn main() {
    let args: Vec<String> = env::args().collect();
    let static_dir = "./static/bigram";
    let device = LibTorchDevice::Cpu;
    if args[1].eq("train") {
        let config = TrainingConfig::new(BigramLanageModelConfig::new(), AdamWConfig::new());
        std::fs::create_dir_all(static_dir).ok();
        config
            .save(format!("{static_dir}/config.json"))
            .expect("Config should be saved successfully");
        B::seed(config.seed);
        let batcher_train = BigramBatcher::<B>::new(device);
        let batcher_valid =
            BigramBatcher::<<Autodiff<LibTorch> as AutodiffBackend>::InnerBackend>::new(device);

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(TextDataset::new("./text.txt", true));

        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(TextDataset::new("./text.txt", false));

        let learner = LearnerBuilder::new(static_dir)
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(CpuUse::new())
            .metric_valid_numeric(CpuUse::new())
            .metric_train_numeric(CpuMemory::new())
            .metric_valid_numeric(CpuMemory::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
                Aggregate::Mean,
                Direction::Lowest,
                Split::Valid,
                StoppingCondition::NoImprovementSince { n_epochs: 1 },
            ))
            .devices(vec![device])
            .num_epochs(config.num_epochs)
            .build(config.model.init(&device), config.optimizer.init(), 1e-4);

        let model_trained = learner.fit(dataloader_train, dataloader_test);
        model_trained
            .save_file(format!("{static_dir}/model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    } else if args[1].eq("generate") {
        let input = args[2..].join(" ");
        let (stoi, itos, _) = get_mappings("./text.txt");
        generate::<B>(static_dir, device, &input, 50, &stoi, &itos);
    }
}
