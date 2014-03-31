from nltk.tokenize import word_tokenize
from nltk.tag.stanford import POSTagger
import sqlite3, nltk.tag

db_path = '/home/dc65/Documents/research/irony/ironate.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

def extract_segments():
	# TODO: fetch segments that have not been tagged 
	segment_ids = [cn[0] for cn in c.execute('select distinct id from irony_commentsegment').fetchall()]
	segments = []
	#i = 0
	for segment_id in segment_ids:
		c.execute("select text from irony_commentsegment where id='%s'" % segment_id)
		segments.append(c.fetchall()[0][0].encode('utf-8').strip())
		#i += 1
		#if i > 10:
		#	break

	return segment_ids, segments

def tag(segments):
	st = POSTagger('/home/dc65/Documents/tools/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger', '/home/dc65/Documents/tools/stanford-postagger-2014-01-04/stanford-postagger-3.3.1.jar')
	tagged = []
	for segment in segments:
		x = ' '.join(nltk.tag.tuple2str(w) for w in st.tag(word_tokenize(segment)))
		tagged.append(x.decode('utf-8'))
	return tagged

# find a spot and insert it
def update(ids, tagged):
	#c.execute('begin transaction')
	for id, tag in zip(ids, tagged):
		#print id, tag.encode('utf-8')
		c.execute('update irony_commentsegment set tag="%s" where id="%s"' % (tag, id))
		c.execute('select tag from irony_commentsegment where id="%s"' % id)
		#print c.fetchall()[0][0].encode('utf-8')
		#print id, tag.encode('utf-8')
	#c.execute('commit')
	conn.commit()

def main():
	# if len(sys.argv) != 2:
	# 	print 'usage: python extract_segments.py output'
	# 	sys.exit(0)
	#c.execute('alter table irony_commentsegment add column tag text')

	ids, segments = extract_segments()
	tagged = tag(segments)
	update(ids, tagged)

main()
