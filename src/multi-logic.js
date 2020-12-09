import * as tf from '@tensorflow/tfjs';

const NUM_LAYERS = 3;
const NUM_UNITS = 25;

const inputs = tf.tensor([
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
]);

const outputs = tf.tensor([
  [ 0, 0, 0 ], [ 0, 1, 1 ], [ 0, 1, 1 ], [ 1, 1, 0 ],
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
  model.add(tf.layers.dense({ units: 3, activation: 'relu' }));

  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  });

  return model;
};

const onEpochEnd = (epoch, info) => {
  if (epoch % 50 === 0) {
    console.log(`Epoch ${epoch} complete. Accurracy: ${info.acc}`);
  }
}

const trainXor = (model) => {
  return model.fit(inputs, outputs, {
    epochs: 500,
    batchSize: 1,
    validationData: [inputs, outputs],
    callbacks: {
      onEpochEnd,
    }
  });
};

const testIt = (model) => {
  const inputs = [[0,0],[0,1],[1,0],[1,1]];
  console.log('Logic Results (input => [and, or, xor]):');
  inputs.map(input => {
    const result = model.predict(tf.tensor([input]));
    console.log(` ${JSON.stringify(input)} => ${JSON.stringify(result.arraySync()[0])}`);
  })
}

const xor = () => {
  const model = createXorModel();
  trainXor(model).then(info => {
    const finalAccurancy = info.history.acc[info.history.acc.length-1]
    console.log(`Final accuracy: ${finalAccurancy}`);
    if (finalAccurancy > 0.99) {
      console.log('Model successfully generated');
    } else {
      console.log('Model not generated successfully!!');
    }
    testIt(model);
  });
};

export default xor;