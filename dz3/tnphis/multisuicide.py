import os
import multiprocessing
import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import csr_matrix, lil_matrix

def write_data(data):
  with open(os.path.abspath(os.path.dirname(__file__) + './out.csv'), 'w') as outf:
    outf.write('id,prob\n')
    for itm in data:
      outf.write(str(int(itm[0])) + ',' + str(float(itm[1])) + '\n')
    outf.close()

def load_text_data(total_pts, labels = True, **args):
  with open(os.path.abspath(os.path.dirname(__file__) + './' + args['filename']), 'r') as td:
    index = -1
    X = np.zeros(
      (total_pts),
      dtype = [('id', 'u2'), ('text', 'S1000')]
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

class CustomTextVectorizer():
  def __init__(self, **args):
    self.data = args['data']
    self.labels = args.get('labels')
    self.stop_words = args.get('stop_words')
    self.fn = args.get('fn')
    self.vecs = None

    if self.fn is None:
      self.fn = 'training_vecs.csv'

    # best so far
    # stop_words = self.stop_words,
    # ngram_range = (2, 3),
    # max_df = .2,
    # max_features = 16000

    self.vectorizer = TfidfVectorizer(
      stop_words = self.stop_words,
      ngram_range = (2, 3),
      max_df = .2,
      max_features = 16000
    )

    self.vectorizer.fit(self.data['text'])
    print(len(self.vectorizer.vocabulary_.items()))

  def vectorize(self, data):
    self.vecs = self.vectorizer.transform(data)

  def write(self):
    with open(os.path.abspath(os.path.dirname(__file__) + './' + self.fn), 'w') as outf:
      outf.write('id,' + ','.join(['f' + str(i) for i in range(len(self.vecs.toarray()[0]))]) + ',label\n')
      for index, itm in enumerate(self.vecs.toarray()):
        current_row = str(self.data['id'][index]) + ',' + ','.join(list(str(f) for f in itm))
        if self.labels is not None:
          current_row += ',' + str(int(self.labels[index]))
        outf.write(current_row + '\n')
      outf.close()

  def dump(self):
    return self.vecs

class StrangeClassifier():
  def __init__(self, **args):
    self.ids = args.get('ids')
    self.X = args['X']
    self.Y = args['Y']
    self.total_training_pts = self.X.shape[0]
    self.total_dims = self.X.shape[1]

    #8366, 16000
    self.structured_data = lil_matrix((self.total_training_pts, self.total_dims + 1))
    print('structured data inited')

    self.structured_data[:, :self.total_dims] = lil_matrix(self.X)
    self.structured_data[:, self.total_dims] = np.reshape(self.Y, (self.total_training_pts, 1))
    print('structured data filled')

    self.sample_size = 3000
    self.current_X = csr_matrix((self.sample_size, self.total_dims))
    self.current_Y = np.zeros((self.sample_size,))
    self.n_iterations = 40

    self.test_data = args['test_data']
    self.total_data_pts = self.test_data.shape[0]

    self.current_estimates = np.zeros((self.total_data_pts, 2))
    self.current_crosscheck_estimates = None
    self.agg_estimates = []
    self.agg_crosscheck_estimates = []

    self.calc_start = dt.datetime.now()

  def shuffle_training_data(self):
    #np.random.shuffle(self.structured_data)

    index = np.arange(np.shape(self.structured_data)[0])
    np.random.shuffle(index)
    self.structured_data =  self.structured_data[index, :]

    self.current_X = self.structured_data[:, :self.total_dims]
    self.current_Y = self.structured_data[:, self.total_dims].toarray().reshape((self.total_training_pts,))

  def estimate(self):
    classifier = BernoulliNB(class_prior = [.9, .1])
    for i in range(self.n_iterations):
      self.shuffle_training_data()
      classifier.fit(self.current_X, self.current_Y)

      results_proba = classifier.predict_proba(self.test_data)
      self.current_estimates[:, 0] = self.ids
      self.current_estimates[:, 1] = results_proba[:, 1]
      #else:
      self.current_crosscheck_estimates = {
        'labels': classifier.predict(self.test_data),
        'proba': results_proba[:, 1]
      }

      self.agg_estimates.append(self.current_estimates)
      self.agg_crosscheck_estimates.append(self.current_crosscheck_estimates)

  def run(self):
    self.estimate()
    #print(self.agg_estimates)
    return self.agg_estimates

  def run_crosscheck(self):
    self.estimate()
    #print(self.agg_crosscheck_estimates)
    return self.agg_crosscheck_estimates

def run_process(X, Y, test_data, ids):
  runner = StrangeClassifier(X = X, Y = Y, test_data = test_data, ids = ids)
  return runner.run()

def run_process_crosscheck(X, Y, test_data):
  runner = StrangeClassifier(X = X, Y = Y, test_data = test_data)
  return runner.run_crosscheck()

if __name__ == '__main__':
  total_data_pts = 13944
  crosscheck = False
  stop_words = ('а',	'бы',	'в',	'во',	'вот',	'для',	'до',	'если',	'же',	'за',	'и', 'из',	'или',	'к',	'ко',	'между',	'на',	'над',	'но',	'о',	'об',	'около',	'от',	'по',	'под',	'при',	'про',	'с',	'среди', 'то',	'у', 'чтобы')
  td = load_text_data(total_data_pts, True, filename='train.csv')
  testd = load_text_data(total_data_pts, False, filename='test_without_labels.csv')

  calc_start = dt.datetime.now()
  myvec = CustomTextVectorizer(
    data = td[0],
    labels = td[1],
    stop_words = stop_words
  )

  myvec.vectorize(td[0]['text'])
  tv = myvec.dump()

  myvec.vectorize(testd['text'])
  testv = myvec.dump()

  X_train, X_test, Y_train, Y_test = train_test_split(tv, td[1], test_size = .4)
  test_data_pts = Y_test.shape[0]

  print('Vectorization time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')

  thread_cnt = multiprocessing.cpu_count() - 1
  pool = multiprocessing.Pool(processes = thread_cnt)

  process_arrays = []
  data_arrays = []
  for cpu in range(thread_cnt):
    if crosscheck:
      process_arrays.append(pool.apply_async(run_process_crosscheck, [
        X_train,
        Y_train,
        X_test,
      ]))
    else:
      process_arrays.append(pool.apply_async(run_process, [
        tv,
        td[1],
        testv,
        testd
      ]))

  for p in process_arrays:
    data_arrays.append(p.get())

  print('Exec time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')

  if crosscheck:
    Y_pred = np.zeros(Y_test.shape)
    Y_proba = np.zeros(Y_test.shape)
    datas = []
    for idx, arr in enumerate(data_arrays):
      datas += arr

    for est in datas:
      #maximum
      for index, l in enumerate(est['labels']):
        if Y_pred[index] < l:
          Y_pred[index] = l

      for index, p in enumerate(est['proba']):
        #Y_proba[index] += p / len(datas)
        if Y_proba[index] < p:
          Y_proba[index] = p

    print(Y_pred[:100], Y_test[:100])
    print(classification_report(y_true = Y_test, y_pred = Y_pred))
    print('roc_auc', roc_auc_score(y_true = Y_test, y_score = Y_proba))
  else:
    final_results = np.zeros((total_data_pts, 2))
    datas = []
    for idx, arr in enumerate(data_arrays):
      datas += arr

    for est in datas:
      for index, itm in enumerate(est):
        final_results[index][0] = itm[0]
        final_results[index][1] += itm[1] / len(datas)

    write_data(final_results)

