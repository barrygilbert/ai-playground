import * as tf from '@tensorflow/tfjs-node';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config();
const IS_DEBUG = process.env.IS_DEBUG === 'true';
const FILE_PATH = process.env.FILE_PATH;

const NUM_LAYERS = 1;
const NUM_UNITS = 15;

const createModel = ({ numLayers = NUM_LAYERS, numUnits = NUM_UNITS }) => {
  const model = tf.sequential();
  for(var i=0;i<numLayers;i++) {
    const opts = { units: numUnits, activation: 'relu' };
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

const trainModel = (name, model, inp, out) => {
  const inputs = tf.tensor(inp);
  const outputs = tf.tensor(out);
  console.log(`${name} Training Started`);
  return model.fit(
    inputs,
    outputs,
    {
      epochs: 25,
      batchSize: 1,
      verbose: IS_DEBUG ? 1 : 0,
      validationData: [inputs, outputs],
      callbacks: {
        onEpochEnd,
      },
    },
  );
};

const testModel = (inputs, model, name) => {
  console.log(`${name} Results:`);
  inputs.map(input => {
    const result = model.predict(tf.tensor([input]));
    console.log(` ${JSON.stringify(input)} => ${result.arraySync()[0]}`);
  })
}

const createAndTrainModel = (name, inputs, outputs, testInputs) => (attempt = 0) => {
  const model = createModel({});
  trainModel(name, model, inputs, outputs).then(info => {
    const finalAccurancy = info.history.acc[info.history.acc.length-1];
    console.log(`${name} Final accuracy: ${finalAccurancy}`);
    testModel(testInputs, model, name);
    if (finalAccurancy > 0.99) {
      console.log(`${name} Model successfully generated`);
      if (FILE_PATH) {
        model.save(`file:///${path.join(FILE_PATH, name)}`);
      }
    } else {
      console.log(`${name} Model not generated successfully!!`);
      if (attempt < 5) {
        console.log(`Attempting to generate ${name} again...`);
        return createAndTrainModel(name, inputs, outputs, testInputs)(attempt + 1);
      }
    }
  });
};

export default createAndTrainModel;