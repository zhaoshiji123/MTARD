import os
import argparse
import torch
from mtard_loss import *
from cifar100_models import *
import torchvision
from torchvision import datasets, transforms

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

prefix = 'resnet18_CIFAR10_MTRAD_'
epochs = 300
batch_size = 128
epsilon = 8/255.0
weight_learn_rate = 0.025
bert = 1
adv_teacher_path = ''
nat_teacher_path = ''

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

student = resnet18()
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

weight = {
    "adv_loss": 1/2.0,
    "nat_loss": 1/2.0,
}
init_loss_nat = None
init_loss_adv = None

def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
teacher = wideresnet()
teacher.load_state_dict(torch.load(adv_teacher_path))
teacher = teacher.cuda()
teacher.eval()

teacher_nat = wideresnet()
teacher_nat.load_state_dict(torch.load(nat_teacher_path))
teacher_nat = teacher_nat.cuda()
teacher_nat.eval()


for epoch in range(1,epochs+1):
    print('the {}th epoch '.format(epoch))
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_nat_logits = teacher_nat(train_batch_data)
        student_adv_logits,teacher_adv_logits = mtard_inner_loss_ce(student,teacher,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        student.train()
        student_nat_logits = student(train_batch_data)
        kl_Loss1 = kl_loss(F.log_softmax(student_adv_logits,dim=1),F.softmax(teacher_adv_logits.detach(),dim=1))
        kl_Loss2 = kl_loss(F.log_softmax(student_nat_logits,dim=1),F.softmax(teacher_nat_logits.detach(),dim=1))
        kl_Loss1 = 10*torch.mean(kl_Loss1)
        kl_Loss2 = 10*torch.mean(kl_Loss2)

        if init_loss_nat == None:
            init_loss_nat = kl_Loss2.item()
        if init_loss_adv == None:
            init_loss_adv = kl_Loss1.item()

        lhat_adv = kl_Loss1.item() / init_loss_adv
        lhat_nat = kl_Loss2.item() / init_loss_nat


        inv_rate_adv = lhat_adv**bert
        inv_rate_nat = lhat_nat**bert


        #weight_learn_rate = 0.025
        weight["nat_loss"] = weight["nat_loss"] - weight_learn_rate *(weight["nat_loss"] - inv_rate_nat/(inv_rate_adv + inv_rate_nat))
        #weight["adv_loss"] = weight["adv_loss"] - weight_learn_rate *(weight["adv_loss"] - inv_rate_adv/(inv_rate_adv + inv_rate_nat))
        weight["adv_loss"] = 1 - weight["nat_loss"] 

        total_loss = weight["adv_loss"]*kl_Loss1 + weight["nat_loss"]*kl_Loss2

        total_loss.backward()
        optimizer.step()

        if step%100 == 0:
            print('weight_nat: ', weight["nat_loss"],'nat_loss: ',kl_Loss2.item(),' weight_adv: ', weight["adv_loss"],' adv_loss: ',kl_Loss1.item())

    if (epoch % 20 == 0 and epoch <215) or (epoch%1 == 0 and epoch >= 215) or epoch == 1:

        test_accs = []
        test_accs_naturals = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            logits = student(test_batch_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs_naturals = test_accs_naturals + predictions.tolist()
        test_accs_naturals = np.array(test_accs_naturals)
        test_accs_natural = np.sum(test_accs_naturals==0)/len(test_accs_naturals)
        print('natural acc',np.sum(test_accs_naturals==0)/len(test_accs_naturals))
        torch.save(student.state_dict(),'./model/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        weight_learn_rate *= 0.1
