from make_dataset import load_dataset
from model import train_model
import click
import torch


@click.command()
@click.option("--epochs", type=int, default=10)
@click.option("--INITIAL_EPOCH", is_flag=True)
@click.option("--STEPS_PER_EPOCH", is_flag=True)
def run(epochs, INITIAL_EPOCH, STEPS_PER_EPOCH):
    try:
        history_1[INITIAL_EPOCH] = model_1.fit(
                x=dataset_train,
                epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                initial_epoch=INITIAL_EPOCH,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback
                ]
            )

            model_name = 'recipe_generation_rnn_raw_' + str(INITIAL_EPOCH) + '.h5'
            model_1.save(model_name, save_format='h5')
    except AttributeError as e:
        print(e)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter