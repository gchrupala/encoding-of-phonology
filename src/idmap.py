import imaginet.data_provider as dp

pro_d = dp.getDataProvider(dataset='coco', root='..')
IDMAP = {}
for split in ['val', 'test', 'train', 'restval']:
    IDMAP[split] = {}
    for i, image in enumerate(pro_d.iterImages(split=split)):
        IDMAP[split][i] = {}
        for j, sent in enumerate(image['sentences']):
            IDMAP[split][i][j] = sent['sentid']
import json
with open('../data/coco/dataset.idmap.json','w') as out:
    json.dump(IDMAP, out)
