import * as tf from '@tensorflow/tfjs-node';
import createModel from './create-model';
import { loadTrainingData, loadTestingData, getTrainingData, getTestingData } from './data';

const trainModel = (model) => {
  //const testingData = loadTestingData();
  return model.fitDataset(
    getTrainingData(),
    {
      epochs: 25,
      validationData: getTestingData(),
      callbacks: [
        tf.callbacks.earlyStopping({
          monitor: 'loss',
          patience: 3,
        })
      ],
    }
  );
  /*
  const trainingData = loadTrainingData();
  return model.fit(
    trainingData.inputs,
    trainingData.outputs,
    {
      batchSize: 100,
      epochs: 50,
    }
  );
  */
};

const cnn = (attempts = 0) => {
  const model = createModel(28, 28, 3);
  trainModel(model).then(info => {
    const finalAccurancy = info.history.categoricalAccuracy[info.history.categoricalAccuracy.length-1];
    console.log(`Final accuracy: ${finalAccurancy}`);
    if (finalAccurancy < 0.99) {
      if (attempts < 10) {
        console.log(`Running again`);
        cnn(attempts + 1);
      }
    } else {
      model.save('file:///home/barry/projects/ai-playground/built/cnn')
    }
  });
};

export default cnn;