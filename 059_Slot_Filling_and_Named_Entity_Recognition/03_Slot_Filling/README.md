## Requirements

- `Pytorch 0.3.0` or newer
- `Python 3.6`
- `Perl`

## Usage

- Training and Prediction

  ```bash
  python main.py [-h] [--train-data-path TRAIN_DATA_PATH]
                 [--test-data-path TEST_DATA_PATH]
                 [--slot-names-path SLOT_NAMES_PATH]
                 [--saved-model-path SAVED_MODEL_PATH]
                 [--result-path RESULT_PATH] [--mode {elman,jordan,hybrid,lstm}]
                 [--bidirectional] [--cuda]
  ```

- [Evaluation](./eval/conlleval.md)

  ```bash
  perl eval/conlleval.pl -d "\t" < data/output.txt
  ```

