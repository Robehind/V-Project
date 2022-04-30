datadir = '/mnt/ssd/vdata/thordata'
wd_path = '/mnt/ssd/vdata/word_embedding/word_embedding.hdf5'
data_name = 'resnet50fc_nn.hdf5'
# tscenes = {'kitchen': '2,3,8,9,11,12,14,15,22-28',
#            'living_room': '2,5,8,10,11,13,19,21-26,28,29',
#            'bedroom': '1,2,4,7,10,12,13,15,16,18-20,22,26,29',
#            'bathroom': '2,5,9,11,13,15,17-20,22,23,25,26,29'}
escenes = {'kitchen': '1,4,20,21,30',
           'living_room': '1,12,14,17,27',
           'bedroom': '3,6,14,17,27',
           'bathroom': '3,4,7,24,27'}
# vscenes = {'kitchen': '5,6,17,19,29',
#            'living_room': '4,6,7,16,20',
#            'bedroom': '5,8,21,24,28',
#            'bathroom': '8,12,14,21,28'}
vscenes = {}
tscenes = {
        'kitchen': '1-6,8,9,11,12,14,15,17,19-30',
        'living_room': '1,2,4-8,10-14,16,17,19-29',
        'bedroom': '1-8,10,12-22,24,26-29',
        'bathroom': '2-5,7-9,11-15,17-29'}
targets = [
    'Fridge', 'Microwave', 'Sink', 'GarbageCan', 'LightSwitch',
    'Sofa', 'Television', 'Bed', 'AlarmClock', 'Laptop',
    'HandTowel', 'SoapBottle']