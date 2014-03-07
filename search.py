import praw

""" Given "nnp" (keyword), "id" (user id), and "n" (# of comments), search1 returns user's comments containing word "nnp." """
def search1(nnp, id, n):
	n = min(n, 100)
	r = praw.Reddit(user_agent="cdg720")
	user = r.get_redditor(id)
	comments_containing_nnp = []
	for comment in user.get_comments(sort='new', time='all', limit=n):
		if nnp in comment.body.split():
			comments_containing_nnp.append(comment)
	return comments_containing_nnp

""" Given "nnp" (keyword), "subreddit" (subreddit), search2 returns comments containing nnp from subreddit "subreddit." """
def search2(nnp, subreddit):
	# TODO: restrict to some user?
	r = praw.Reddit(user_agent="cdg720")
	comments_containing_nnp = []
	for post in r.get_subreddit(subreddit).search(nnp):
		for comment in post.comments:
			# TODO: sometime it returns praw.objects.MoreComments. why?
			if type(comment) is praw.objects.Comment and nnp in comment.body.split():
				comments_containing_nnp.append(comment)
	return comments_containing_nnp

#TODO: If we want more comments, expand morecomments.
""" Given nnp and url, search returns comments containing nnp from a post indicated by url. """
def search3(nnp, url):
	r = praw.Reddit(user_agent='cdg720')
	post = r.get_submission(url)
	for comment in post._comments:
		if type(comment) is praw.objects.Comment and nnp in comment.body.split():
			yield comment

""" search 4 is same as search3 except that it takes a set of nnps. """
def search4(nnps, url):
	r = praw.Reddit(user_agent='cdg720')
	post = r.get_submission(url)
	for comment in post._comments:
		if type(comment) is praw.objects.Comment and bool(nnps & set(comment.body.split())):
			yield comment
			
def main():
	nnps = set()
	nnps.add('this')
	nnps.add('what')
	for com in search4(nnps, 'http://www.reddit.com/r/funny/comments/1zncub/i_put_a_halloween_mask_on_my_dog/'):
		print com
main()
