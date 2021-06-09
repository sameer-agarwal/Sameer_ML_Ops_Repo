import sys

sys.path.insert(1,r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\04_Continuous_Integration\Structure_Example_repo\src\data')
from make_dataset import main

def check_size(images):
    for i in range(len(images)):
        assert images[i].shape == (1,28,28)


def test_data():

    batch_size, train_set, test_set = main()
    j = 0
    #Checking the training set
    for images, labels in train_set:
        check_size(images)
        j = j + 1

    Total_len = j*batch_size
    assert Total_len == 60032

    #Checking the testing set
    i = 0

    for images, labels in test_set:

        check_size(images)
        i = i + 1

    Total_len = i*batch_size
    assert Total_len == 10048

if __name__ == '__main__':
    test_data()

    
