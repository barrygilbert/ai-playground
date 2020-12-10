import * as tf from '@tensorflow/tfjs-node';

const createModel = (width, height, colors) => {
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [width, height, colors] }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPool2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPool2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
  }));
  model.add(tf.layers.dense({
    units: 10,
  }));

  model.compile({
    optimizer: 'adam',
    metrics: ['categoricalAccuracy'],
    loss: tf.losses.softmaxCrossEntropy,
  });

  return model;
};

export default createModel;