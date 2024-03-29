def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def print_simple_progress_bar(progress):
    """
    Call in a loop to create terminal progress bar
    @params:
        progress   - Required  : total progress out of 100 (Int)
    """
    print('\r[{0}] {1}%\r'.format('#' * (int(progress / 10)), progress), end="")


def console_log(arg):
    """
    Log argument and its type to console, surrounding by borders
    @params:
        arg   - Required  : Argument to log
    """
    print('-------------------------')
    print(arg)
    print('type is ', type(arg))
    print('-------------------------')
