import * as tf from '@tensorflow/tfjs';

const inputs = tf.tensor([
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
  [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ],
]);

const outputs = tf.tensor([
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
  [ 0 ], [ 1 ], [ 1 ], [ 0 ],
]);

const createXorModel = () => {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [2], units: 15, activation: 'relu' }),
      tf.layers.dense({ units: 15, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'relu' })
    ]
  });

  model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  });

  return model;
};

const trainXor = (model) => {
  return model.fit(inputs, outputs, {
    epochs: 50,
    batchSize: 1,
  });
};

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
    // model.predict(tf.tensor([[0,0],[0,1],[1,0],[1,1]])).print();
  });
};

export default xor;