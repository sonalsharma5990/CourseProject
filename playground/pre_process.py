"""This module filter the documents from NYTimes corpus."""
from datetime import datetime
import tarfile
from lxml import etree

NY_CORPUS = '../data/nyt_corpus_LDC2008T19.tgz'

def get_dates(from_date, to_date):
    """Get Year Month date from start to end date."""
    start_date = datetime.strptime(from_date,'%Y%m')
    end_date = datetime.strptime(to_date, '%Y%m')
    dates = [start_date.strftime('%Y/%m')]
    while start_date < end_date:
        month, year = start_date.month, start_date.year
        if month == 12:
            start_date = datetime(year+1, 1, 1)
        else:
            start_date = datetime(year, month+1, 1)
        dates.append(start_date.strftime('%Y/%m'))
    return dates
        


def get_text(doc_file, keywords=None):
    """Extract text from article."""
    


def filter_doc(from_date, to_date, keywords=None):
    tar = tarfile.open(NY_CORPUS)
    for date_ in get_dates(from_date, to_date):
        monthly_tar = tarfile.open(
            fileobj=tar.extractfile(f'nyt_corpus/data/{date_}.tgz'))
        doc_paths = [member.name 
            for member in monthly_tar
            if member.isfile()]
        # sort paths by dates
        doc_paths = sorted(doc_paths)
        for doc_path in doc_paths:
            doc_file = monthly_tar.extractfile(doc_path)
            get_text(doc_file, keywords)




if __name__ == '__main__':
    # filter_doc(None, None)
    filter_doc('200005','200010')