import json
from collections import Counter
from tqdm import tqdm
import pickle


def load_as_triple(kb_json):
    '''
    Get triples from the kb.json
    Not ignore the repeat triples
    '''
    triples = []
    vocab = {'<PAD>': 0,
             '<UNK>': 1,
             '<START>': 2,
             '<END>': 3}
    print("Build triples and vocabulary of kb")
    kb = json.load(open(kb_json))
    print("Process the concepts...")
    for i in tqdm(kb['concepts']):
        for j in kb['concepts'][i]['instanceOf']:
            s = kb['concepts'][i]['name']
            o = kb['concepts'][j]['name']
            triples.append([s, 'instanceOf', o])
    print("Process the entities...")
    for i in tqdm(kb['entities']):
        for j in kb['entities'][i]['instanceOf']:
            s = kb['entities'][i]['name']
            o = kb['concepts'][j]['name']
            triples.append([s, 'instanceOf', o])
        
        name = kb['entities'][i]['name']
        for attr_dict in kb['entities'][i]['attributes']:
            o = '{}_{}'.format(attr_dict['value']['value'], attr_dict['value'].get('unit', ''))
            triples.append([name, attr_dict['key'], o])
            s = '{}_{}_{}'.format(name, attr_dict['key'], o)
            for qk, qvs in attr_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{}_{}'.format(qv['value'], qv.get('unit', ''))
                    triples.append([s, qk, o])

        for rel_dict in kb['entities'][i]['relations']:
            o = kb['entities'].get(rel_dict['object'], kb['concepts'].get(rel_dict['object'], None))
            if o is None:
                continue
            o = o['name']
            if rel_dict['direction'] == 'backward':
                triples.append([o, rel_dict['predicate'], name])
            else:
                triples.append([name, rel_dict['predicate'], o])
            s = '{}_{}_{}'.format(name, rel_dict['predicate'], o)
            for qk, qvs in rel_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{}_{}'.format(qv['value'], qv.get('unit', ''))
                    triples.append([s, qk, o])
    print("Completed, the length of triples is {}".format(len(triples)))
    return triples

if __name__ == "__main__":
    triples = load_as_triple("data/kb.json")
    with open("data/kb_triple_list.json","wb") as f:
        pickle.dump(triples,f)
