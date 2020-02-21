// Import template file.

local template = import "template_fei.libsonnet";

////////////////////

// Set options.

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "ner",

  // If debugging, don't load expensive embedding files.
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: false,
  tune_glove: false,
  use_char: false,
  use_elmo: false,
  elmo_option: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
  elmo_weight: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
  // elmo_option: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
  // elmo_weight: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
  use_attentive_span_extractor: false,
  use_bert_base: true,
  use_bert_large: false,
  finetune_bert: true,
  rel_prop: 0,
  coref_prop: 0,
  context_width: 1,
  rel_prop_dropout_A: 0.0,
  rel_prop_dropout_f: 0.0,

  // Specifies the model parameters.
  use_lstm: false,
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  lstm_dropout: 0.0,
  feature_size: 20,
  feedforward_layers: 2,
  char_n_filters: 50,
  feedforward_dim: 150,
  max_span_width: 8,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  loss_weights: {          // Loss weights for the modules.
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 0.0,
    span: 0,
    seq: 0,
  },
  loss_weights_events: {   // Loss weights for trigger and argument ID in events.
    trigger: 0.0,
    arguments: 0.0,
  },

  // Coref settings.
  coref_spans_per_word: 0.4,
  coref_max_antecedents: 100,

  // Relation settings.
  relation_spans_per_word: 0.5,
  relation_positive_label_weight: 1.0,

  // Event settings.
  trigger_spans_per_word: 0.4,
  argument_spans_per_word: 0.8,
  events_positive_label_weight: 1.0,

  // Model training
  batch_size: 2,
  instances_per_epoch: 1000,
  num_epochs: 250,
  patience: 15,
  optimizer: {
    type: "bert_adam",
    lr: 1e-3,
    warmup: 0.1,
    t_total: 200000,
    weight_decay: 0.0,
    parameter_groups: [
      [["_text_field_embedder"], {"type": "bert_adam", "lr": 5e-5, "warmup": 0.2, "t_total": 200000, "weight_decay": 0.01}],
    ],
  },
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 4
  },
  // add by feili
  shuffle: true,
  evaluate_on_test: true,
  // endpoint, pooling, conv, attention, rnn
  span_extractor: "endpoint",
  combination: "x,y",
  // model: dygie, cls_ner, seq_ner, tree_ner
  model: "tree_ner",
  // seq_ner: flat, stacked
  label_scheme: "stacked",
  // use tree info or not
  use_tree: false,
  tree_prop: 1,
  use_syntax: false,
  tree_dropout: 0,
  // pooling, conv, attention, rnn
  tree_children: 'attention',
  tree_span_filter: false,
  tree_match_filter: false,
  tree_feature_first: true,
  // scibert_scivocab_cased, multi_cased_L-12_H-768_A-12
  bert_name: "scibert_scivocab_cased",
  // tree feature, transformer
  use_tree_feature: true,
  // 'transformer', 'gcn'
  tree_feature_arch: 'transformer',
  tft_layers: 1,
  tft_head: 48,
  tft_kv: 16,
  tft_dropout: 0.1,
  tree_feature_dim: 48,
  tree_feature_layer: 1,
  tree_feature_dropout: 0.1,
  tree_feature_dict: ['F1'],
  // add, concat
  tree_feature_usage: 'concat',
};

////////////////////

// Feed options into template.

template(params)
