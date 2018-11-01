# -*- coding: utf-8 -*-
import pandas as pd
import git
import os
import logging
import datetime


# use logger
logger = logging.getLogger(__name__)


def get_project_dir():
        '''search for git repo
        '''
        logger.debug('get project dir')
        repo = git.Repo('.', search_parent_directories=True)
        return repo.working_tree_dir


def get_project_path(file_path):
        ''' return path concat with project dir
        '''
        logger.debug('get project path')
        return os.path.join(get_project_dir(), file_path)


def setmax_rows(i):
        '''
        set maximum number of rows displayed

        Parameters
        ---------------
        i : int
            number of rows to be displayed
        '''
        pd.options.display.max_rows = i


def setmax_columns(i):
        '''
        set maximum number of column displayed

        Parameters
        ---------------
        i : int
            number of column to be displayed
        '''
        pd.options.display.max_columns = i


def concat_str(x):
        return "{%s}" % ', '.join(x)


def date_from_yearweek(date_str):
        '''Return date from year week

        Parameters
        ----------
        date_str: date like string
                e.g. "2017-W50"
        '''
        logging.info("Date from {}".format(date_str))
        date = pd.to_datetime(datetime.datetime.strptime(date_str + "-2",
                                                         "%Y-W%U-%w"))
        logging.info("Date to {}".format(date))
        return date


def add_YYYY_WW_starting_TUE(df, year_week_only=False, remove_date=False):
        '''Add year_week starting from Tuesday to dataframe

        Parameters:
        df: dataframe
                needs date
        '''
        df = df.copy()
        logging.debug('to datetime')
        df['date'] = pd.to_datetime(df['date'])
        df['weekday_name'] = df.date.dt.weekday_name

        logging.debug('strftime')
        df['year_week'] = df.date.dt.strftime("%Y_%W")
        df['year'] = df['year_week'].str.slice(0, 4)
        df['week'] = df['year_week'].str.slice(5, 7).astype(int)

        # removing one day will make the week start from Tuesday
        logging.debug('remove one from monday')
        df.loc[(df.weekday_name == "Monday"), 'week'] = df.week - 1
        df['week'] = df.week.astype(str)

        logging.debug('pad')
        df['week'] = df.week.str.pad(2, side='left', fillchar='0')

        logging.debug('new year week')
        df['year_week'] = df['year'] + "_" + df['week']

        # remove edge cases
        # for 2016, 2016_52 and 2017_00 are in the same week
        df.loc[df['year_week'] == '2017_00', 'year_week'] = '2016_52'

        # remove if year week only
        if year_week_only:
                del df['year'], df['week'], df['weekday_name']
        if remove_date:
                del df['date']
        logging.debug('added year_week')
        return df
