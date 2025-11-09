from execute_util import link


def named_link(name: str, url: str) -> str:
    return link(title=f" [{name}]", url=url)


def article_link(url: str) -> str:
    return link(title=" [article]", url=url)


def blog_link(url: str) -> str:
    return link(title=" [blog]", url=url)


def x_link(url: str) -> str:
    return link(title=" [X]", url=url)


def youtube_link(url: str) -> str:
    return link(title=" [video]", url=url)