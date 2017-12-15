import pymorphy2
import os
import multiprocessing
import datetime as dt
import numpy as np
import re

stop_words = [
  'а',
  'ах',
  'бы',
  'быть',
  'в',
  'во',
  'вот',
  'всего',
  'всё',
  'вы',
  'для',
  'да',
  'до',
  'если',
  'ещё',
  'же',
  'за',
  'и',
  'из',
  'ие',
  'ия',
  'или',
  'к',
  'ки',
  'как',
  'ко',
  'который',
  'кто',
  'ку',
  'ли',
  'лишь',
  'между',
  'мы',
  'на',
  'над',
  'нибудь',
  'никак',
  'но',
  'ну'
  'о',
  'об',
  'ой',
  'ок',
  'около',
  'он',
  'она',
  'они',
  'от',
  'по',
  'под',
  'практически',
  'при',
  'про',
  'просто',
  'с',
  'совсем',
  'среди',
  'ство',
  'так',
  'таки',
  'тем',
  'то',
  'ты',
  'ть',
  'тот',
  'у',
  'уже',
  'чем',
  'что',
  'чтобы',
  'это',
  'этот',
  'src',
  'https',
  'figure'
]

def string_to_ngram(string, n):
  for item in string.split():
    for i in range(len(item) - n + 1):
      yield item[i:i + n]

def write_data(data, labels = None):
  fn = 'test_normalized.csv'
  if labels is not None:
    fn = 'train_normalized.csv'
  with open(os.path.abspath(os.path.dirname(__file__) + './' + fn), 'w') as outf:
    outf.write('id,test,label\n')
    for index, itm in enumerate(data):
      currentline = str(data['id'][index]) + ',' + data['text'][index].decode('utf8').replace('\n', '')
      if labels is not None:
        currentline += ',' + str(int(labels[index]))
      currentline += '\n'
      outf.write(currentline)
    outf.close()

def normalize_text(txt):
  myan = pymorphy2.MorphAnalyzer()
  newtxt = ''
  newpos = ''
  decoded_txt = txt.decode('utf8').lower()
  for w in decoded_txt.split(' '):
    myword = myan.parse(w)
    if w not in stop_words:
      newtxt += re.sub(r'[^\w\s]', '', myword[0].normal_form) + ' '
      if myword[0].tag and myword[0].tag.POS:
        newpos += ' ' + myword[0].tag.POS
    if 'таблетка' in newtxt and 'не болезнь' in newtxt:
      print(myword, ' : ', w, ' : ', newtxt, ' : ', decoded_txt)

  newtxt = newtxt.replace(' не ', ' не_')
  if newtxt[:3] == 'не ':
    newtxt = 'не_' + newtxt[3:]

  #newtxt = re.sub(r'\s', '', newtxt)
  newgrams = ''
  for w in newtxt.split(' '):
    newgrams += ' ' + ' '.join(string_to_ngram(w, 4))

  newtxt = newpos
  # newtxt = newgrams

  return newtxt.strip().encode('utf8')

def normalize_array(data):
  for idx, txt in enumerate(data):
    data[idx] = normalize_text(txt)
  return data

def load_text_data(total_pts, labels = True, **args):
  with open(os.path.abspath(os.path.dirname(__file__) + './' + args['filename']), 'r') as td:
    index = -1
    X = np.zeros(
      (total_pts),
      dtype = [('id', 'u2'), ('text', 'S16000')]
    )
    if labels:
      Y = np.zeros(total_pts)

    for line in td:
      sl = line.split(',')
      if index > -1 and index < total_pts:
        X[index] = (sl[0], sl[1].encode('utf8'))
        if labels:
          Y[index] = sl[2]

      index += 1

    td.close()

  if labels:
    return (X, Y)
  else:
    return X

def run_normalizer(vec):
  return normalize_array(vec)

def normalize_parallel(data):
  total_data_pts = len(data)
  thread_cnt = multiprocessing.cpu_count() - 1
  pool = multiprocessing.Pool(processes = thread_cnt)
  process_arrays = []
  data_arrays = []

  for cpu in range(thread_cnt):
    start_index = cpu * total_data_pts // thread_cnt
    end_index = (cpu + 1) * total_data_pts // thread_cnt
    process_arrays.append(pool.apply_async(run_normalizer, [data[start_index:end_index]]))

  for p in process_arrays:
    data_arrays.append(p.get())

  for idx, arr in enumerate(data_arrays):
    data[idx * total_data_pts // thread_cnt:(idx + 1) * total_data_pts // thread_cnt] = arr

  return data


if __name__ == '__main__':
  total_data_pts = 13944
  td = load_text_data(total_data_pts, True, filename='train.csv')
  #testd = load_text_data(total_data_pts, False, filename='test_without_labels.csv')

  calc_start = dt.datetime.now()
  #testd['text'] = normalize_parallel(testd['text'])
  td[0]['text'] = normalize_parallel(td[0]['text'])
  print('Normalization time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')
  write_data(td[0], td[1])
  #write_data(testd)
