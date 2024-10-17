import emotion_clf.ml as ml
from emotion_clf.ml.inference_utils import *
import glob
import tqdm


class MER():
    def __init__(self,
                 pretrain_dir,
                 ):
        
        if os.path.isfile(pretrain_dir):  # if is single model
            self.experiment_dir = [pretrain_dir]
        else:  # if is ensemble
            self.experiment_dir = glob.glob(pretrain_dir)
    
    def predict(self, generated, return_logits=False):
        probs = []
        print("Loading MER...")
        for path in tqdm.tqdm(self.experiment_dir):  # all models
            completed_file_path = os.path.join(path, "completed")
            if not os.path.exists(completed_file_path):
                print(f"Can't find completed model of {path}")

            exp = ml.loading.load_experiment(experiment_dir=path,
                                            data_dir=generated)
            out_probs = exp.evaluate("inference",
                            restore_file_name="last",
                            use_swa=exp.params.get("swa") is not None,
                            save_predictions=True)

            # Collecting all outputs from each model
            probs.append(out_probs.numpy())
        logits = np.array(probs)
        top_3_indices = np.argsort(logits.mean(axis=0))[-3:]  # ascending
        binarized = np.zeros_like(logits.mean(axis=0))
        binarized[top_3_indices] = 1
        out_tags = []
        for i in np.flip(top_3_indices):
            out_tags.append(TAG_MAP[str(i)])
        if return_logits:
            return out_tags, logits
        return out_tags


def main():
    checkpoint_path = 'path_to_checkpoint'
    mermodel = MER(checkpoint_path)
    out_tags = mermodel.predict('../../emotion/audio/1.mp3')
    print(out_tags)

if __name__ == "__main__":
    main()

# python inference.py --experiment_dir ./experiments/convs-m128* --data_dir ../emotion/audio/1.mp3
# python inference.py --experiment_dir ./experiments/filters-m128* --data_dir ../emotion/audio/1.mp3