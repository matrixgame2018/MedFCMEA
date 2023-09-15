import pandas as pd

data = pd.read_csv('F:/MedFMC/MedFMC_train/chest/train.csv', encoding='utf-8')
print(data.values)
with open('chest_trainval.txt','a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0]) + ' ' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + ',' + str(line[4]) + ',' +
                     str(line[5]) + ',' +str(line[6]) + ',' +str(line[7]) + ',' +str(line[8]) + ',' +
                     str(line[9]) + ',' + str(line[10]) + ',' +str(line[11]) + ',' +str(line[12]) + ',' + str(line[13]) + ',' +
                     str(line[14]) + ',' +str(line[15]) + ',' +str(line[16]) + ',' + str(line[17]) + ',' +
                     str(line[18]) + ',' + str(line[19]) + '\n'))
#
data = pd.read_csv('F:/MedFMC/MedFMC_train/colon/train.csv', encoding='utf-8')
with open('colon_trainval.txt','a+', encoding='utf-8') as f:
     for line in data.values:
         f.write((str(line[0]) + ' ' + str(line[1]) + '\n'))

data = pd.read_csv('F:/MedFMC/MedFMC_train/endo/train.csv', encoding='utf-8')
print(data.values)
with open('endo_trainval.txt','a+', encoding='utf-8') as f:
     for line in data.values:
        f.write((str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str(line[4]) + '\n'))


val_list = ['endo']
for i in range(len(val_list)):
     data = pd.read_csv('F:/MedFMC/MedFMC_val/'+val_list[i]+'/test_witout_label.csv', encoding='utf-8')
     with open(val_list[i]+'_test_without_label.txt','a+', encoding='utf-8') as f:
         for line in data.values:
             print(line)
             f.write((str(line[0])+'\n'))