import os
#os.mkdir('../Experimento_3/Resultados_ssd')
#os.mkdir('../Experimento_3/Resultados_ssd/ssd512')
#os.mkdir('../Experimento_3/Resultados_ssd/ssd300')
#os.mkdir('../Experimento_3/Resultados_ssd/ssd7')

print ('Training ssd7')
os.system('python train.py -c config_7.json')
print ('Testing ssd7')
os.system('python train.py -c config_7.json')


print ('Training ssd300')
os.system('python train.py -c config_300.json')
print ('Testing ssd300')
os.system('python train.py -c config_7.json')
