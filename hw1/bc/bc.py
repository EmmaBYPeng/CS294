import argparse
import tensorflow as tf
from proc_data import Data
from bc_model import BC


def create_model(session, params):
    model = BC(optimizer=params["optimizer"], lr=params["lr"])
    session.run(tf.initialize_all_variables())
    return model


def train(params):
    pass


def overfit(params):
    num_small_data = 100
    data = Data(params["data_file"], train_ratio=0.9, val_ratio=0.05)
    small_data = data.get_small_dataset(num_data=num_small_data)

    batch_size, num_epochs = params["batch_size"], params["num_epochs"]
    num_batch_per_epoch = int((num_small_data - 1) / batch_size) + 1

    with tf.Session() as sess:
        model = create_model(sess, params)
        batches = data.batch_iter(small_data, batch_size, num_epochs)
        for i, (batch_x, batch_y) in enumerate(batches):
            pred, loss = model.step(sess, batch_x, batch_y)
            print("Epoch %04d step %08d loss %04f" % (i/num_batch_per_epoch, i, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="../data/Hopper_data.pkl")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--overfit", type=bool, default=True)
    args = vars(parser.parse_args())

    if args["overfit"]:
        overfit(args)
    elif args["train"]:
        train(args)
