import praw


''' global '''
MAX_COMMENTS = 100
MAX_POST_COMMENTS = 5 # max comments we'll grab per a given post
USER_ID= "byron"
r = praw.Reddit(user_agent="%s" % USER_ID)


def _find_nnp_in(nnp, comments):
    comments_containing_nnp = []
    for comment in comments:
        if nnp in comment.body.split():
            comments_containing_nnp.append(comment)
    return comments_containing_nnp


'''
given "nnp" (keyword), "id" (user id), and "n" (# of comments), 
returns user comments containing token "nnp."
'''
def search_comments_by_user(nnp, target_user_id, n=MAX_COMMENTS):
    n = min(n, MAX_COMMENTS)
    user = r.get_redditor(target_user_id)
    comments = user.get_comments(sort='new', time='all', limit=n)
    return _find_nnp_in(nnp, comments)

'''
Given "nnp" (keyword), "subreddit" (subreddit), 
return comments containing nnp from subreddit "subreddit."
'''
def search_subreddit_comments(nnp, subreddit):
    # TODO: restrict to some user?
    r = praw.Reddit(user_agent="cdg720")
    comments = []
    for post in r.get_subreddit(subreddit).search(nnp, limit=MAX_COMMENTS):
        for comment in post.comments[:MAX_POST_COMMENTS]:
            if type(comment) is praw.objects.Comment:
                comments.append(comment)

    return _find_nnp_in(nnp, comments)
