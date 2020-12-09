import * as tf from '@tensorflow/tfjs-node';
import { fileSystem } from '@tensorflow/tfjs-node/dist/io';
import dotenv from 'dotenv';
dotenv.config();
const IS_DEBUG = process.env.IS_DEBUG === 'true';
const FILENAME = process.env.FILENAME;

const NUM_LAYERS = 1;
const NUM_UNITS = 15;

const inputs = tf.tensor([
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
]);

const outputs = tf.tensor([
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
]);

const createXorModel = () => {
  const model = tf.sequential();
  for(var i=0;i<NUM_LAYERS;i++) {
    const opts = { units: NUM_UNITS, activation: 'relu' };
    if (i === 0) {
      opts.inputShape = [2];
    }
    model.add(tf.layers.dense(opts));
  }
  model.add(tf.layers.dense({ units: 1, activation: 'relu' }));

  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  });

  return model;
};

const onEpochEnd = (epoch, info) => {
  if (epoch % 50 === 0 && IS_DEBUG) {
    console.log(`Epoch ${epoch} complete. Accurracy: ${info.acc}`);
  }
}

const trainXor = (model) => {
  return model.fit(inputs, outputs, {
    epochs: 150,
    batchSize: 1,
    verbose: IS_DEBUG ? 1 : 0,
    validationData: [inputs, outputs],
    callbacks: {
      onEpochEnd,
    }
  });
};

const testXor = (model) => {
  const inputs = [[0,0],[0,1],[1,0],[1,1]];
  console.log('XOR Results:');
  inputs.map(input => {
    const result = model.predict(tf.tensor([input]));
    console.log(` ${JSON.stringify(input)} => ${result.arraySync()[0]}`);
  })
}

const xor = (attempt = 0) => {
  const model = createXorModel();
  trainXor(model).then(info => {
    const finalAccurancy = info.history.acc[info.history.acc.length-1];
    if (IS_DEBUG) {
      console.log(`Final accuracy: ${finalAccurancy}`);
      testXor(model);
    }
    if (finalAccurancy > 0.99) {
      console.log('Model successfully generated');
      if (FILENAME) {
        model.save(FILENAME);
      }
    } else {
      console.log('Model not generated successfully!!');
      if (attempt < 5) {
        console.log('Attempting to generate xor again...');
        return xor(attempt + 1);
      }
    }
  });
};

export default xor;