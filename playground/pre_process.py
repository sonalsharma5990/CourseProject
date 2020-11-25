"""This module filter the documents from NYTimes corpus."""
from datetime import date, datetime
import os
import tarfile
import logging
import shutil
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

from lxml import etree

NY_CORPUS = '../data/nyt_corpus_LDC2008T19.tgz'
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
    level=logging.INFO)



def get_date_range(from_date, to_date):
    """Get Year Month date range from start to end date."""
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
        


def get_text(doc_file, keywords=None, include_heading=False):
    """Extract text from article."""
    tree = etree.parse(doc_file)
    all_paras = tree.xpath("/nitf/body/body.content/block[@class='full_text']/p")
    content = None
    if keywords:
        content = ' '.join(para.text 
            for para in all_paras
            if any(word in para.text for word in keywords))
    else:
        content = ' '.join(para.text 
            for para in all_paras)

    if include_heading:
        heading = tree.xpath('/nitf/body/body.head/hedline/hl1')
        content += heading[0].text
    # replace any newline with spaces
    content = content.replace('\n',' ')
    return content

def extract_date(monthly_extract, date_, doc_paths, keywords, out_dir):
    """Extract content for single date and save in file."""
    with tarfile.open(monthly_extract) as monthly_tar:
        with open(f'{out_dir}/{date_}.txt','w') as out_file:
            for doc_path in doc_paths:
                with monthly_tar.extractfile(doc_path) as doc_file:
                    content = get_text(doc_file, keywords)
                if content:
                    out_file.write(content +'\n')
                    logger.info('Added %s %s',date_,content[:128])


def create_date_map(monthly_tar, date_):
    """Get day of month and document path mapping."""
    doc_date_map = {}
    doc_paths = [member.name 
            for member in monthly_tar
            if member.isfile()]
        # sort paths by dates
    doc_paths = sorted(doc_paths)
    for doc_path in doc_paths:
        doc_date = date_.replace('/','') + doc_path[3:5]
        if doc_date in doc_date_map:
            doc_date_map[doc_date].append(doc_path)
        else:
            doc_date_map[doc_date] = [doc_path]
    return doc_date_map

def filter_doc(from_date, to_date, out_dir, keywords=None):
    """Extract documents from NYTimes corpus tar.gz file."""
    tar = tarfile.open(NY_CORPUS)
    dates = []
    for date_ in get_date_range(from_date, to_date):
        tar.extract(f'nyt_corpus/data/{date_}.tgz',out_dir)
        out_extract = f'{out_dir}/nyt_corpus/data/{date_}.tgz'
        with tarfile.open(out_extract) as monthly_tar:
            doc_date_map = create_date_map(monthly_tar, date_)
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(
                extract_date,
                out_extract, date_, doc_paths, keywords, out_dir)
            for date_, doc_paths in doc_date_map.items()]
            for f in as_completed(futures):
                f.result()
        
        dates.extend(doc_date_map.keys())

    shutil.rmtree(f'{out_dir}/nyt_corpus')
    tar.close()
    return dates

def extract_doc(from_date, to_date, out_dir, keywords=None):
    """Extract documents for range from_date to_date."""
    dates = filter_doc(
            from_date,to_date,out_dir ,
            keywords=keywords)
    combine_files(out_dir, dates)
    remove_files(out_dir, dates)

    

def remove_files(out_dir, dates):
    for date_ in dates:
        os.remove(f'{out_dir}/{date_}.txt')
        logger.info('Deleted file for date %s', date_)


def combine_files(out_dir, dates):
    """Combine documents for each date."""
    doc_data_map = {}
    doc_index = 0
    with gzip.open(f'{out_dir}/data.txt.gz', 'wb') as data_file:
        for date_ in dates:
            logger.debug('Processing file for date %s', date_)
            with open(f'{out_dir}/{date_}.txt') as f:
                for line in f:
                    if not line.strip():
                        logger.error('Empty document %s: %s', doc_index, line)
                    
                    data_file.write(line.encode('utf-8'))          
                    doc_data_map[doc_index] = date_
                    doc_index += 1
    with open(f'{out_dir}/doc_date_map.txt','w') as f:
        for doc_id,date_ in doc_data_map.items():
            f.write('{},{}\n'.format(doc_id, date_,))
    logger.info('Number of documents %s', doc_index)





if __name__ == '__main__':
    # filter_doc(None, None)
    out_dir = 'experiment_1'
    # extract_doc(
    #         '200005','200010',out_dir ,
    #         keywords=['Bush','Gore'])
    out_dir = 'experiment_2'
    # extract_doc(
    #         '200007','200112',out_dir)

