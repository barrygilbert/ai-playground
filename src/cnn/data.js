import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';

const TRAINING_IMAGES = 'train-images-idx3-ubyte';
const TRAINING_LABELS = 'train-labels-idx1-ubyte';

const TESTING_IMAGES = 't10k-images-idx3-ubyte';
const TESTING_LABELS = 't10k-labels-idx1-ubyte';

const calcInt = (arr) => {
  return arr[0] * 16777216 + arr[1] * 65536 + arr[2] * 256 + arr[3];
}

const loadDataGenerator = (images, labels, max = -1) => {
  const imageData = [...fs.readFileSync(images)];
  const labelData = [...fs.readFileSync(labels)];

  const count = calcInt(imageData.slice(4, 8));
  const width = calcInt(imageData.slice(8, 12));
  const height = calcInt(imageData.slice(12, 16));
  const size = width * height;

  function* GetInput() {
    let i = 0;
    while (i < count && (max === -1 || i < max)) {
      const input = tf.tensor(imageData.slice(i*size+16, i*size+16+size).map(v => {
        const a = v / 255;
        return [a, a, a];
      })).reshape([width, height, 3]);
      i++;
      yield input;
    }
  }

  function* GetOutput() {
    let i = 0;
    while (i < count && (max === -1 || i < max)) {
      const output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      const v = labelData[8+i];
      output[v] = 1;
      i++;
      yield output;
    }
  }

  const xs = tf.data.generator(GetInput);
  const ys = tf.data.generator(GetOutput);
  const ds = tf.data.zip({ xs, ys }).batch(32);
  return ds;
};

const loadData = (images, labels) => {
  const imageData = [...fs.readFileSync(images)];
  const labelData = [...fs.readFileSync(labels)];

  const count = calcInt(imageData.slice(4, 8));
  const width = calcInt(imageData.slice(8, 12));
  const height = calcInt(imageData.slice(12, 16));

  const inputs = imageData.slice(16).map(v => {
    const a = v / 255;
    return [a, a, a];
  });
  const outputs = labelData.slice(8).map(v => {
    const res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    res[v] = 1;
    return res;
  });

  return {
    inputs: tf.tensor(inputs, [count, width, height, 3]),
    outputs: tf.tensor(outputs),
  };
};

export const loadTrainingData = () => {
  return loadData(TRAINING_IMAGES, TRAINING_LABELS);
};

export const loadTestingData = () => {
  return loadData(TESTING_IMAGES, TESTING_LABELS);
};

export const getTrainingData = (max = -1) => {
  return loadDataGenerator(TRAINING_IMAGES, TRAINING_LABELS, max);
};

export const getTestingData = (max = -1) => {
  return loadDataGenerator(TESTING_IMAGES, TESTING_LABELS, max);
};