# download_abide_struc_civet_vertex.py
#
# Author: Daniel Clark, 2015
# Updated to python 3 and to support downloading by DX, Cameron Craddock, 2019

"""
This script downloads data from the Preprocessed Connetomes Project's
ABIDE Preprocessed data release and stores the files in a local
directory; users specify derivative, pipeline, strategy, and optionally
age ranges, sex, site of interest

Usage:
    python download_abide_struc_civet_vertex.py -d <derivative> -p <pipeline>
                                     -s <strategy> -o <out_dir>
                                     [-lt <less_than>] [-gt <greater_than>]
                                     [-x <sex>] [-t <site>]
"""


# 数据下载程序
def collect_and_download(derivative, pipeline, out_dir, less_than, greater_than, site, sex, diagnosis):
    """

    Function to collect and download images from the ABIDE preprocessed
    directory on FCP-INDI's S3 bucket

    Parameters
    ----------
    derivative : string
        surfaces or measure of interest
    pipeline : string
        pipeline used to process data of interest
    out_dir : string
        filepath to a local directory to save files to
    less_than : float
        upper age (years) threshold for participants of interest
    greater_than : float
        lower age (years) threshold for participants of interest
    site : string
        acquisition site of interest
    sex : string
        'M' or 'F' to indicate whether to download male or female data
    diagnosis : string
        'asd', 'tdc', or 'both' corresponding to the diagnosis of the
        participants for whom data should be downloaded

    Returns
    -------
    None
        this function does not return a value; it downloads data from
        S3 to a local directory

    :param surfaces:
    :param pipeline:
    :param out_dir:
    :param less_than: 
    :param greater_than: 
    :param site: 
    :param sex:
    :param diagnosis:
    :return: 
    """

    # Import packages
    import os
    import urllib.request as request

    # 平均逐帧位移的阈值
    mean_fd_thresh = 0.2
    # 初始化数据下载地址
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
    s3_pheno_path = os.path.abspath("../../data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv")

    # 获取必要的工具类型、处理流程类型和策略
    derivative = derivative.lower()

    # 检查是否包含ROI，如果包含则保存后缀为'.1D'，否则就为'.nii.gz'
    extension = '.txt'

    # If output path doesn't exist, create it
    if not os.path.exists(out_dir):
        print('Could not find {0}, creating now...'.format(out_dir))
        os.makedirs(out_dir)

    # 读取表型数据
    s3_pheno_file = open("../../data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv", "r")
    pheno_list = s3_pheno_file.readlines()
    print(pheno_list[0])

    # 得到数据表的表头
    header = pheno_list[0].split(',')
    try:
        site_idx = header.index('SITE_ID')
        file_idx = header.index('FILE_ID')
        age_idx = header.index('AGE_AT_SCAN')
        sex_idx = header.index('SEX')
        dx_idx = header.index('DX_GROUP')
        mean_fd_idx = header.index('func_mean_fd')
    except Exception as exc:
        err_msg = 'Unable to extract header information from the pheno file: {0}\nHeader should have pheno info:' \
                  ' {1}\nError: {2}'.format(s3_pheno_path, str(header), exc)
        raise Exception(err_msg)

    # 通过表型文件构建完整的下载地址
    print('Collecting images of interest...')
    s3_paths = []
    for pheno_row in pheno_list[1:]:

        # 获取每一行数据
        cs_row = pheno_row.split(',')

        try:
            # 获取文件ID
            row_file_id = cs_row[file_idx]
            # 获取对应的实验室
            row_site = cs_row[site_idx]
            # 获取对应的年龄
            row_age = float(cs_row[age_idx])
            # 获取对应的性别
            row_sex = cs_row[sex_idx]
            # 获取是ASD还是TDC
            row_dx = cs_row[dx_idx]
            # 获取平均逐帧位移
            row_mean_fd = float(cs_row[mean_fd_idx])
        except Exception as e:
            err_msg = 'Error extracting info from phenotypic file, skipping...'
            print(err_msg)
            continue

        # 假如为指定文件名，那么就跳过
        if row_file_id == 'no_filename':
            continue
        # 如果平均逐帧位移fd的太大，那么就跳过
        if row_mean_fd >= mean_fd_thresh:
            continue

        # Test phenotypic criteria (three if's looks cleaner than one long if)
        # Test sex
        if (sex == 'M' and row_sex != '1') or (sex == 'F' and row_sex != '2'):
            continue

        if (diagnosis == 'asd' and row_dx != '1') or (diagnosis == 'tdc' and row_dx != '2'):
            continue

        # Test site
        if site is not None and site.lower() != row_site.lower():
            continue
        # Test age range
        if greater_than < row_age < less_than:
            filename = row_file_id + '_' + derivative + extension
            s3_path = '/'.join([s3_prefix, 'Outputs', pipeline, 'surfaces_' + derivative, filename])
            print('Adding {0} to download queue...'.format(s3_path))
            s3_paths.append(s3_path)
        else:
            continue

    # 下载数据
    total_num_files = len(s3_paths)
    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix).split("/")[-1]
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        try:
            if not os.path.exists(download_file):
                print('Retrieving: {0}'.format(download_file))
                request.urlretrieve(s3_path, download_file)
                print('{0:3f}% percent complete'.format(100*(float(path_idx+1)/total_num_files)))
            else:
                print('File {0} already exists, skipping...'.format(download_file))
        except Exception as exc:
            print('There was a problem downloading {0}.\n Check input arguments and try again.'.format(s3_path))

    # Print all done
    print('Done!')


# Make module executable
if __name__ == '__main__':

    # Import packages
    import argparse
    import os
    import sys

    # 初始化参数解析
    parser = argparse.ArgumentParser(description=__doc__)

    # 必选参数
    # -a表示只下载ASD患者，-c表示只下载对照组的被试者，-a -c表示都下载
    parser.add_argument('-a', '--asd', required=False, default=False, action='store_true',
                        help='Only download data for participants with ASD.'
                             ' Specifying neither or both -a and -c will download data from all participants.')
    parser.add_argument('-c', '--tdc', required=False, default=False, action='store_true',
                        help='Only download data for participants who are typically developing controls.'
                             ' Specifying neither or both -a and -c will download data from all participants.')

    # 可选参数
    # 小于某个年龄段
    parser.add_argument('-lt', '--less_than', nargs=1, required=False,
                        type=float, help='Upper age threshold (in years) of participants to download (e.g. for '
                                         'subjects 30 or younger, \'-lt 31\')')
    # 大于某个年龄
    parser.add_argument('-gt', '--greater_than', nargs=1, required=False,
                        type=int, help='Lower age threshold (in years) of participants to download (e.g. for '
                                       'subjects 31 or older, \'-gt 30\')')
    # 设置指定的站点
    parser.add_argument('-t', '--site', nargs=1, required=False, type=str,
                        help='Site of interest to download from (e.g. \'Caltech\'')
    # 设置指定的性别
    parser.add_argument('-x', '--sex', nargs=1, required=False, type=str,
                        help='Participant sex of interest to download only (e.g. \'M\' or \'F\')')

    # 解析参数
    args = parser.parse_args()

    # 初始化可选参数

    # 如果设置了下载ASD和TDC，或都没有设置，那么就全都下载
    desired_diagnosis = ''
    if args.tdc == args.asd:
        desired_diagnosis = 'both'
        print('Downloading data for ASD and TDC participants')
    elif args.tdc:
        desired_diagnosis = 'tdc'
        print('Downloading data for TDC participants')
    elif args.asd:
        desired_diagnosis = 'asd'
        print('Downloading data for ASD participants')

    try:
        desired_age_max = args.less_than[0]
        print('Using upper age threshold of {0:d}...'.format(desired_age_max))
    except TypeError:
        desired_age_max = 200.0
        print('No upper age threshold specified')

    try:
        desired_age_min = args.greater_than[0]
        print('Using lower age threshold of {0:d}...'.format(desired_age_min))
    except TypeError:
        desired_age_min = -1.0
        print('No lower age threshold specified')

    try:
        desired_site = args.site[0]
    except TypeError:
        desired_site = None
        print('No site specified, using all sites...')

    try:
        desired_sex = args.sex[0].upper()
        if desired_sex == 'M':
            print('Downloading only male subjects...')
        elif desired_sex == 'F':
            print('Downloading only female subjects...')
        else:
            print('Please specify \'M\' or \'F\' for sex and try again')
            sys.exit()
    except TypeError:
        desired_sex = None
        print('No sex specified, using all sexes...')

    # functional_data_dir = os.path.abspath("../../data/ABIDE/functionals/cpac/filt_global/")
    save_data_dir = os.path.abspath("../../data/ABIDE/data/structurals/")

    # 获取流程类别，1个参数中的一个：ants
    desired_pipeline = 'civet'

    # 表面的类型，6个参数获取选取不定数量
    # ['mid_surface_rsl_left_native_area_40mm', 'mid_surface_rsl_right_native_area_40mm',
    # 'native_pos_rsl_asym_hemi', 'surface_rsl_left_native_volume_40mm', 'surface_rsl_right_native_volume_40mm']
    desired_derivative_list = ['mid_surface_rsl_left_native_area_40mm']

    for desired_derivative in desired_derivative_list:
        # 构建保存地址
        output_data_dir = os.path.join(save_data_dir, desired_pipeline, 'vertex', desired_derivative)

        # 下载数据
        collect_and_download(desired_derivative, desired_pipeline, output_data_dir, desired_age_max,
                             desired_age_min, desired_site, desired_sex, desired_diagnosis)
