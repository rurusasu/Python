from functools import reduce

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    """
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


if __name__ == "__main__":
    from collections import namedtuple
    LineItem = namedtuple('LineItem', ['商品ID', '単価', '数量'])
    
    def result(items):
        return compose(items)

    line_items = [
        LineItem('apple', 250, 5),
        LineItem('banana', 500, 9),
        LineItem('Orange', 50, 4),
    ]
    
    print(result(line_items))
