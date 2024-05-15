import numpy as np
import pandas as pd
import torch.nn as nn
import torch, random, os, cv2, argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


seed_num = 42
def seed_worker(worker_id):
    np.random.seed(seed_num)
    random.seed(seed_num)

g = torch.Generator()
g.manual_seed(seed_num)



def preprocess_images(images):
    """
    args:
    images: a list of images
    
    return:
    an array of resized images of desirable size
    
    """
    TARGET_SIZE = (36, 36)
    images_ = []
    for img in images:
        image = cv2.resize(img/img.max(), dsize=(TARGET_SIZE[0], TARGET_SIZE[1]), interpolation=cv2.INTER_CUBIC)
        images_.append(image)
        
    return np.asarray(images_).astype("float32")
    
    
def AugmentedImages(fold, img):
    """
    args:
    fold: Number of images
    img: An image
    
    return:
    augmented images
    """
    aug_images = []

    while fold != len(np.unique(aug_images, axis=0)):
        angle = np.random.randint(0, 361)
        aug_images.append(Transformations(angle, img))
    
    len_aug_img = len(np.unique(aug_images, axis=0))
    
    assert fold == len_aug_img
    
    return np.unique(aug_images, axis=0)
    
    
def Transformations(angle, img):
    """
    args:
    angle: An angle to rotate an image
    img: An image
    
    return:
    A transformed image via bijective functions
    
    In the context of transforming an image, a bijective function refers to a mapping between two sets of pixels (before and after the transformation) where every pixel in the source         image is uniquely paired with a pixel in the target image, and vice versa. This ensures that the transformation is both injective (one-to-one) and surjective (onto), meaning that no      two pixels in the source image map to the same pixel in the target image, and every pixel in the target image corresponds to exactly one pixel in the source image.
    
    """
    
    t1 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(angle), transforms.RandomHorizontalFlip(),])
    t2 = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(angle),])
    t3 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(0.25), ])
    t4 = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(0.65), transforms.RandomAffine(angle)])
    t5 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomAffine(angle)])
    t6 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomHorizontalFlip(0.45), transforms.RandomAffine(angle)])
    t7 = transforms.Compose([transforms.ToTensor(),transforms.RandomRotation(angle), transforms.RandomVerticalFlip(0.2), transforms.RandomAffine(angle)])
    t8 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.35), transforms.RandomAffine(angle)])
    t9 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.05),])
    t10 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.25),transforms.RandomRotation(angle), ])
    t11 = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.25), transforms.RandomRotation(angle), transforms.RandomVerticalFlip(0.75),])
    t12 = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(0.15), transforms.RandomHorizontalFlip(0.45)])
    t13 = transforms.Compose([transforms.ToTensor(), transforms.RandomAffine(angle),])
    
    transformations = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]
    
    return transformations[np.random.randint(0, 13)](img).numpy()


def test_accuracy(network, test_loader, criterion, device):
    """
    args:
    network: Wafer2Spike, an SNN model
    test_loader: Data loader for test set
    criterion: Crossentropy Loss
    device: CPU or CUDA (GPU)
    
    """
    
    test_loss = 0
    correct = 0
    network.eval()
    for batch_id, (data, target) in enumerate(test_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        test_loss += loss.to('cpu').item()
        correct += sum(np.argmax(output.data.cpu().numpy(), 1) == target.data.cpu().numpy())

    print("Test loss: {:.6f} | Test accuracy: {:.6f}".format(test_loss / len(test_loader), correct / len(test_loader.dataset)))
                                                                  
                                                                  
    
class Wafer_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data)
        



def training(network, params, batch_size=64, epochs=10, lr=0.0001, dataloaders=None, numClasses=9, spike_ts=10, dropout_fc=0.3, model_type="Wafer2Spike_4C"):
                      
    """
    args:
    network: Current-based spiking neural network: Wafer2Spike
    params: (SCDECAY, VDECAY, VTH, CW); parameters for each LIF neuron
    batch_size: Number of images in one batch
    epochs: Number of epochs for training
    lr: Learning rate
    dataloaders: Loading and preprocessing of data from a dataset into the training, validation, or testing pipelines
    numClasses: Number of classes
    spike_ts: Number of timesteps
    dropout_fc: Dropout percentage for spiking-based fully connected layers
    model_type: Type of the models: Wafer2Spike_2C | Wafer2Spike_3C | Wafer2Spike_4C
    
    """
    
    if len(dataloaders)==3:
        train_loader, val_loader, test_loader = dataloaders
    elif len(dataloaders)==2:
        train_loader, test_loader = dataloaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiating Wafer2Spike Network
    wafer2spike_snn = network(numClasses, dropout_fc, spike_ts, device, params=params)
    wafer2spike_snn = nn.DataParallel(wafer2spike_snn.to(device))
    
    # Instantiating a loss function
    criterion = nn.CrossEntropyLoss()

    # Both synaptic current and voltage decay for each spiking layer
    decays = ['module.wafer2spike.conv_spk_enc_w_vdecay', 'module.wafer2spike.conv_spk_enc_w_scdecay', 'module.wafer2spike.Spk_conv1_w_vdecay',
          'module.wafer2spike.Spk_conv1_w_scdecay', 'module.wafer2spike.Spk_conv2_w_vdecay', 'module.wafer2spike.Spk_conv2_w_scdecay', 
          'module.wafer2spike.Spk_conv3_w_vdecay', 'module.wafer2spike.Spk_conv3_w_scdecay', 'module.wafer2spike.Spk_conv4_w_vdecay',
          'module.wafer2spike.Spk_conv4_w_scdecay', 'module.wafer2spike.Spk_fc_w_vdecay', 'module.wafer2spike.Spk_fc_w_scdecay']

    weights_ts = ['module.wafer2spike.w_t']
    
    decay_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in decays, wafer2spike_snn.named_parameters()))))
    ts_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in weights_ts, wafer2spike_snn.named_parameters()))))
    
    weights = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in decays+weights_ts, wafer2spike_snn.named_parameters()))))

    # Instantiating an optimizer
    optimizer = torch.optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr}, {'params': ts_params, 'lr': lr}], lr=lr)
    
    
    # Training 
    for e in range(epochs):
        loss_per_epoch = 0
        correct = 0
        wafer2spike_snn.train()
        for i, data in enumerate(train_loader, 0):
            wafer_data, label = data
            label = label.type(torch.LongTensor)
            wafer_data, label = wafer_data.to(device), label.to(device)
            optimizer.zero_grad()
            output = wafer2spike_snn(wafer_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.to('cpu').item()
            correct += sum(np.argmax(output.data.cpu().numpy(), 1) == label.data.cpu().numpy())
        
        for param in decay_params:
            param.data = param.data.clamp(min=1e-7)
            
        print(f"Epoch: {e} | Training Loss: {loss_per_epoch / len(train_loader.dataset)} | Training accuracy : {correct / len(train_loader.dataset)}")
        test_accuracy(wafer2spike_snn, test_loader, criterion, device)
        
        
    pred = []
    y = []
    wafer2spike_snn.eval()
    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pred += np.argmax(wafer2spike_snn(data).data.cpu().numpy(), 1).tolist()
        y += target.data.cpu().numpy().tolist()
    
    # Confusion Matrix
    print()
    print("Confusion Matrix :")
    print(confusion_matrix(y, pred))
    print()
    print("Classification Report :")
    print(classification_report(y, pred))
              
    
    name = model_type
    PATH = os.getcwd()+"/"+"Models/" 
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    torch.save(wafer2spike_snn, PATH + name +".pt")
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Wafer Map Pattern Classification.')
    parser.add_argument('--vTh', type = float, help = "Threshold voltage", default = 0.1)
    parser.add_argument('--scDecay', type = float, help = "Synaptic current decay", default = 0.1)
    parser.add_argument('--vDecay', type = float, help = "Voltage decay", default = 0.1)
    parser.add_argument('--thrWin', type = float, help = "Threshold window for neurons", default = 0.3)
    parser.add_argument('--epoch', type = int, help = "Number of epochs.", default = 20)
    parser.add_argument('--batchSize', type = int, help = "Batch Size.", default = 256)
    parser.add_argument('--spikeTs', type = int, help = "Spike time steps.", default = 5)
    parser.add_argument('--learningRate', type = float, help = "Learning Rate.", default = 0.0001) 
    parser.add_argument('--modelType', type = str, help = "Type of the architecture: Wafer2Spike_2C or Wafer2Spike_3C or Wafer2Spike_4C", default = "Wafer2Spike_4C")
    parser.add_argument('--dropout_fc', type = float, help = "Dropout percentage in fully connected layers.", default = 0.3)
    parser.add_argument('--splitRatio', type = str, help = "Split ratio like 8:2 or 8:1:1 or 7:3 or 6:1:3", default = '8:2')
    args = parser.parse_args()
    
    # Fetching values from user's input
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batchSize
    LEARNING_RATE = args.learningRate
    SPLITRATIO = args.splitRatio
    VTH = args.vTh
    SCDECAY = args.scDecay
    VDECAY = args.vDecay
    CW = args.thrWin
    DROPOUT_FC = args.dropout_fc
    MODEL_TYPE = args.modelType

    
    # Training hyperparameters
    N_CLASSES =  9
    SPIKE_TS = 10
    PARAMS = [SCDECAY, VDECAY, VTH, CW]

    if MODEL_TYPE.lower() == "Wafer2Spike_4C".lower():
        from Wafer2Spike_4C import CurrentBasedSNN
    elif MODEL_TYPE.lower() == "Wafer2Spike_3C".lower():
        from Wafer2Spike_3C import CurrentBasedSNN 
    elif MODEL_TYPE.lower() == "Wafer2Spike_2C".lower():
        from Wafer2Spike_2C import CurrentBasedSNN
    else:
        print("Model type does not exist, choose the correct model.")
        assert MODEL_TYPE.lower() in ["Wafer2Spike_2C".lower(), "Wafer2Spike_3C".lower(), "Wafer2Spike_4C".lower()]


    SNN_NETWORK = CurrentBasedSNN

    
    # For reproducibility 
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True
    
    # Loading Wafer Dataset
    print("Loading WM-811k Wafer Data ...", "\n")
    df = pd.read_pickle("../../Wafer_Classification/LSWMD.pkl")

    trte = []
    for j in df["trianTestLabel"]:
      try:
        trte.append(j[0][0])
      except:
        trte.append(np.nan)
    df["trianTestLabel"] = trte
    
    
    ft = []
    for j in df["failureType"]:
      try:
        ft.append(j[0][0])
      except:
        ft.append(np.nan)
    df["failureType"] = ft
    
    
    """
    Mapping :
    
    'Center':0, 'Donut':1, 'Edge-Loc':2, 'Edge-Ring':3, 'Loc':4, 'Random':5, 'Scratch':6, 'Near-full':7, 'none':8
    
    'Training':0,'Test':1
    
    """
    
    df['failureNum'] = df.failureType
    df['trainTestNum'] = df.trianTestLabel
    mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    mapping_traintest = {'Training':0,'Test':1}
    df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})
    
    df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)].reset_index(drop=True)
    df_withlabel['failureNum'] = df_withlabel['failureNum'].astype("int")
    
    # Wafer data
    wafer_data = df_withlabel[df_withlabel["trainTestNum"]==0].reset_index(drop=True)

    # Random sampling without replacement
    wafer_data = wafer_data.sample(n=len(wafer_data), replace=False, random_state=1) # random_state=1 
    
    print("SPLITRATIO:", SPLITRATIO)
    X_train, X_test, y_train, y_test = [], [], [], []
    
    if (SPLITRATIO == '8:2') or (SPLITRATIO == '8:1:1'):
        cls_map_num = {0:2767, 1:329, 2:1958, 3:6802, 4:1311, 5:498, 6:413, 7:49, 8:29357}
    elif SPLITRATIO == '7:3':
        cls_map_num = {0:2489, 1:296, 2:1762, 3:6122, 4:1180, 5:448, 6:372, 7:46, 8:25400}
    elif SPLITRATIO == '6:1:3':
        cls_map_num = {0:2213, 1:263, 2:1566, 3:5443, 4:1049, 5:398, 6:331, 7:39, 8:23486}
            
    for cls in range(len(cls_map_num)):
      X_cls = wafer_data[wafer_data['failureNum'] == cls]
    
      for x,y in zip(X_cls["waferMap"].values[:cls_map_num[cls]], X_cls["failureNum"].values[:cls_map_num[cls]]):
        X_train.append(x)
        y_train.append(y)
    
      for x,y in zip(X_cls["waferMap"].values[cls_map_num[cls]:], X_cls["failureNum"].values[cls_map_num[cls]:]):
        X_test.append(x)
        y_test.append(y)
        
    
    """
    Augmenting same number of samples as mentioned in below paper.
    
    M. B. Alawieh, D. Boning, and D. Z. Pan, “Wafer map defect patterns
    classification using deep selective learning,” in Proc. ACM/IEEE Design
    Automation Conf. (DAC), 2020, pp. 1–6.
    """
    AUGMENTATION = True
    if AUGMENTATION:
        
        print("Augmentation:", bool(AUGMENTATION))
        print("Generating augmented samples ...", "\n")
        
        aug_map_cls_fold =  {'0':2, '1':23, '2':4, '4':8, '5':16, '6':20, '7':166}
        aug_map_cls_oprnd = {'0':7, '1':284, '2':-122, '4':-135, '5':-184, '6':-273, '7':-19}
        
        if (SPLITRATIO == '8:2') or (SPLITRATIO == '8:1:1'):
            aug_map_cls_added_samples = {'0':5541, '1':7851, '2':7710, '3':3884, '4':10353, '5':7784, '6':7987, '7':8115}
        elif SPLITRATIO == '7:3':
            aug_map_cls_added_samples = {'0':4986, '1':7065, '2':6939, '3':3495, '4':9317, '5':7005, '6':7188, '7':7303}
        elif SPLITRATIO == '6:1:3':
            aug_map_cls_added_samples = {'0':4433, '1':6281, '2':6168, '3':3108, '4':8283, '5':6228, '6':6389, '7':6491}
        
        X_train_aug = []
        y_train_aug = []
        
        indices_cls_3 = np.where(np.array(y_train) == 3)[0]
        X_aug, y_aug = [], []
        np.random.shuffle(indices_cls_3)
        
        for ix in indices_cls_3:
          X_train_aug.append(X_train[ix])
          y_train_aug.append(3)
        
        for idx in indices_cls_3[:aug_map_cls_added_samples['3']]:
          img = X_train[idx]
          aug_images = AugmentedImages(1, img)
          for aug_img in aug_images:
            X_aug.append(aug_img.squeeze())
            y_aug.append(3)
        
        
        X_train_aug += X_aug
        y_train_aug += y_aug
        
        for cls in aug_map_cls_fold:
        
          indices_cls = np.where(np.array(y_train) == int(cls))[0]
          fold = aug_map_cls_fold[cls]
          operand = aug_map_cls_oprnd[cls]
          
          X_aug, y_aug = [], []
          for idx in indices_cls:
            img = X_train[idx]
            aug_images = AugmentedImages(fold, img)
            
            for aug_img in aug_images:
              X_aug.append(aug_img.squeeze())
              y_aug.append(int(cls))
        
            X_aug.append(img)
            y_aug.append(int(cls))
        
          if operand > 0:
            np.random.shuffle(indices_cls)
            for idx in indices_cls[:operand]:
              img = X_train[idx]
              aug_images = AugmentedImages(1, img)
              for aug_img in aug_images:
                X_aug.append(aug_img.squeeze())
                y_aug.append(int(cls))
          else:
            X_aug = X_aug[:operand]
            y_aug = y_aug[:operand]
        
          X_train_aug += X_aug
          y_train_aug += y_aug
        
        indices_cls_8 = np.where(np.array(y_train) == 8)[0]
        for ix in indices_cls_8:
          X_train_aug.append(X_train[ix])
          y_train_aug.append(8)
        
        y_train_aug = np.array(y_train_aug)
        
        y_train_aug_index = np.random.permutation(len(y_train_aug))
        wafer_tr_data = preprocess_images(X_train_aug)
        wafer_tr_data = wafer_tr_data[y_train_aug_index]
        wafer_tr_label = y_train_aug[y_train_aug_index]
        wafer_tr_label = torch.tensor(wafer_tr_label).float()
    
    else:
        y_train_ran_index = np.random.permutation(len(y_train))
        wafer_tr_data = preprocess_images(X_train)
        wafer_tr_data = wafer_tr_data[y_train_ran_index]
        wafer_tr_label = np.array(y_train)[y_train_ran_index]
        wafer_tr_label = torch.tensor(wafer_tr_label).float()
        
            
    wafer_test_data = preprocess_images(X_test)
    wafer_test_label = np.array(y_test)
    wafer_test_label = torch.tensor(wafer_test_label).float()
    
    dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2999,), (0.19235,)) ])
    
    if (SPLITRATIO == '8:1:1'):
        wafer_test_data, wafer_val_data, wafer_test_label, wafer_val_label = train_test_split(wafer_test_data, wafer_test_label, test_size=0.50, random_state=42)
        wafer_val_label = torch.tensor(wafer_val_label).float()
        wafer_val_data = dataset_transform(wafer_val_data).permute(1,0,2)
        wafer_val_data = wafer_val_data[:, None, :, :]
        print("Validation data :", wafer_val_data.shape)
        
    elif (SPLITRATIO == '6:1:3'):
        wafer_test_data, wafer_val_data, wafer_test_label, wafer_val_label = train_test_split(wafer_test_data, wafer_test_label, test_size=0.17, random_state=42)
        wafer_val_label = torch.tensor(wafer_val_label).float()
        wafer_val_data = dataset_transform(wafer_val_data).permute(1,0,2)
        wafer_val_data = wafer_val_data[:, None, :, :]
        print("Validation data :", wafer_val_data.shape)
        
    
    wafer_test_label = torch.tensor(wafer_test_label).float()
    wafer_tr_data = dataset_transform(wafer_tr_data).permute(1,0,2)
    wafer_test_data = dataset_transform(wafer_test_data).permute(1,0,2)
    wafer_tr_data = wafer_tr_data[:, None, :, :]
    wafer_test_data = wafer_test_data[:, None, :, :]
    
    print("Training data :", wafer_tr_data.shape, "Test data :", wafer_test_data.shape)
    print()
    
    train_loader = DataLoader(dataset = Wafer_dataset(wafer_tr_data, wafer_tr_label), batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
    
    if (SPLITRATIO == '8:1:1') or (SPLITRATIO == '6:1:3'):
        val_loader = DataLoader(dataset = Wafer_dataset(wafer_val_data, wafer_val_label), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data, wafer_test_label), batch_size=BATCH_SIZE, shuffle=False)
        
    else:
        test_loader = DataLoader(dataset = Wafer_dataset(wafer_test_data, wafer_test_label), batch_size=BATCH_SIZE, shuffle=False)

    
    if (SPLITRATIO == '8:1:1') or (SPLITRATIO == '6:1:3'):
        dataloaders = (train_loader, val_loader, test_loader)
    else:
        dataloaders = (train_loader, test_loader)
    
    
    training(network=SNN_NETWORK, params=PARAMS, batch_size=BATCH_SIZE, epochs=N_EPOCHS, lr=LEARNING_RATE, dataloaders=dataloaders, \
     numClasses=N_CLASSES, spike_ts=SPIKE_TS, dropout_fc=DROPOUT_FC, model_type=MODEL_TYPE)

