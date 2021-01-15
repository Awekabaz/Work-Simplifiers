import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms as trs

registers_file = input('Name of the file with registers: ')
users_file = input('Name of the file with users: ')
added = input('Name of the file with added to live: ')
checkins= input('Name of the file with checkins: ')

registers = pd.read_csv('{}.csv'.format(registers_file))
users = pd.read_csv('{}.csv'.format(users_file))
added = pd.read_csv('{}.csv'.format(added))
checkins = pd.read_csv('{}.csv'.format(checkins))

registers['startTime'] = pd.to_datetime(registers['startTime'])
registers['completionTime'] = pd.to_datetime(registers['completionTime'])

registers['startTime'] = pd.to_datetime(registers['startTime'])
registers['completionTime'] = pd.to_datetime(registers['completionTime'])

cutoff = []
for i, r in registers.iterrows():
    if(pd.isnull(registers.loc[i, 'startTime']) | pd.isnull(registers.loc[i, 'completionTime'])):
        cutoff.append(pd.NaT)
    else:
        diff = registers.loc[i, 'completionTime'] - registers.loc[i, 'startTime']
        cutoff.append(diff)
registers['cutoffTime'] = cutoff
registers.drop(['userIdId', 'courseIdId', 'cutOffTime'],
                                    inplace = True, axis = 1)


trav = added['userId'].to_list()
livestreams = []
for q in registers['userId'].to_list():
    temp = 0
    for qq in trav:
        if(qq == q):
            temp += 1
    livestreams.append(temp)

userId = registers.userId.to_list()
courseId = registers.courseId.to_list()
nickname = []
email = []
first_name = []
last_name = []
mobile = []
mobileHW = []
mobileOS = []
gender = []
for i in userId:
    for index, row in users.iterrows():
        if(i == users.loc[index,'id']):
            nickname.append(users.loc[index, 'nickname'])
            email.append(users.loc[index, 'email'])
            first_name.append(users.loc[index, 'firstName'])
            last_name.append(users.loc[index, 'lastName'])
            mobile.append(users.loc[index, 'mobileNumber'])
            mobileHW.append(users.loc[index, 'mobileHW'])
            mobileOS.append(users.loc[index, 'mobileOS'])


registers['livestreams'] = livestreams
registers['nickname'] = nickname
registers['email'] = email
registers['first_name'] = first_name
registers['last_name'] = last_name
registers['mobile'] = mobile
registers['mobileHW'] = mobileHW
registers['mobileOS'] = mobileOS

def change(x):
    x = x.split('-')[0]
    return x
temp = registers['genderAgeGroup'].to_list()
gender = list(map(change, temp))

registers.insert(3, 'gender', gender)

medals = registers.copy()
medals = medals[medals['racerStatus'] == 'FINISH']
medals.drop(['id','userId', 'registerTime', 'racerStatus', 'startTime', 'completionTime',
'livestreams', 'email', 'mobile', 'mobileHW', 'mobileOS'],
                                    inplace = True, axis = 1)

medals = medals.sort_values(by=['cutoffTime']).reset_index(drop = True)
ranking = []
ran = 0
for X in range(medals.shape[0]):
    ran+=1
    ranking.append(ran)
medals.insert(1, 'overallRanking', ranking)
temp = medals.pop('nickname')
medals.insert(2, 'nickname', temp)

male_rank = 0
female_rank = 0
gender_ranking = []
for ind3, r3 in medals.iterrows():
    if(medals.loc[ind3, 'gender'] == 'Male'):
        male_rank += 1
        gender_ranking.append(male_rank)
    else:
        female_rank += 1
        gender_ranking.append(female_rank)

medals.insert(5, 'genderRanking', gender_ranking)

age_groups = list(set(medals['genderAgeGroup']))
age_group_ran = [0] * len(age_groups)
ageGroupRanking = []
for ind4, r4 in medals.iterrows():
    age_group_ran[age_groups.index(medals.loc[ind4, 'genderAgeGroup'])] += 1
    ageGroupRanking.append(age_group_ran[age_groups.index(medals.loc[ind4, 'genderAgeGroup'])])
medals.insert(7, 'genderAgeGroup_Ranking', ageGroupRanking)

temp = medals.pop('cutoffTime')
medals.insert(2, 'Finish Time', temp)
temp = medals.pop('first_name')
medals.insert(4, 'First Name', temp)
temp = medals.pop('last_name')
medals.insert(5, 'Last Name', temp)
medals.columns = ['Course','Ranking','Finish Time', 'Nickname','First Name','Last Name',
 'Gender', 'BIB Number', 'Gender Ranking', 'Gender Age Group', 'Gender Age Group Ranking']

registers['cutoffTime'] = registers['cutoffTime'].astype(str)
medals['Finish Time'] = medals['Finish Time'].astype(str)
t = registers['cutoffTime'].to_list()
q = medals['Finish Time'].to_list()
cut_temp = []
for tt in t:
    if(tt == 'NaT'):
        cut_temp.append('NaT')
    else:
        tem = tt.split(' ')[2].split('.')[0]
        cut_temp.append(tem)
registers['cutoffTime'] = cut_temp

cut_temp = []
for qq in q:
    if(qq != 'NaT'):
        tem = qq.split(' ')[2].split('.')[0]
        cut_temp.append(tem)
    else:
        cut_temp.append(qq)
medals['Finish Time'] = cut_temp

max_beacons = checkins['sequenceOrder'].max()
register_list = registers['id'].to_list()
max_name = 'ST{}'.format(checkins['sequenceOrder'].max())
for col in range(int(checkins['sequenceOrder'].max())):
    column_name = 'ST{}'.format(col+1)
    temp = []
    temp_ex = []
    for reg in register_list:
        flag = True
        for check, record in checkins[checkins['registerId'] == reg].iterrows():
            if(checkins.loc[check, 'sequenceOrder'] == (col+1) ):
                temp.append(checkins.loc[check, 'checkInTime'])
                temp_ex.append(1)
                flag = False
                break
        if(flag):
            temp.append(pd.NaT)
            temp_ex.append(0)
    if(col == 0):
        registers['Start ST'] = temp
        registers['Start ST-DISC'] = temp_ex
    elif(col == checkins['sequenceOrder'].max() - 1):
        registers['Finish ST'] = temp
        registers['Finish ST-DISC'] = temp_ex
    else:
        registers[column_name] = temp
        registers[column_name+'-DISC'] = temp_ex



des_col = list(filter(lambda k: '-DISC' in k, list(registers)))
registers['overall'] = registers[des_col].sum(axis=1)
registers['stats(%)'] = round( ( registers[des_col].sum(axis=1) / checkins['sequenceOrder'].max() ) * 100, 1)

intervals = registers.copy()
intervals = intervals[intervals['racerStatus'] == 'FINISH']
nicknames = intervals['nickname'].to_list()
bib = intervals['bibNumber'].to_list()
overall1 = intervals['overall'].to_list()
mHW1 =  intervals['mobileHW'].to_list()
mOS1 =  intervals['mobileOS'].to_list()
stats1 =  intervals['stats(%)'].to_list()
data = {'nickname': nicknames, 'BIB Number': bib,'mobileHW': mHW1,
'mobileOS':mOS1, 'overall': overall1, 'stats(%)': stats1}
user_det = pd.DataFrame(data)

user_det['interval'] = ''
for ind1, r1 in user_det.iterrows():
    comp = user_det.loc[ind1, 'stats(%)']
    if(comp>=0.0 and comp< 20.0): user_det.loc[ind1, 'interval'] = '0-20'
    elif(comp>= 20.0 and comp< 40.0): user_det.loc[ind1, 'interval'] = '20-40'
    elif(comp>= 40.0 and comp< 60.0): user_det.loc[ind1, 'interval'] = '40-60'
    elif(comp>= 60.0 and comp< 80.0): user_det.loc[ind1, 'interval'] = '60-80'
    else: user_det.loc[ind1, 'interval'] = '80-100'

if(medals.shape[0] == 0):
    course_name = input('Enter course name: ')
    with pd.ExcelWriter('super_list{}.xlsx'.format(course_name)) as writer:
        registers.to_excel(writer, sheet_name='Full')
        user_det.to_excel(writer, sheet_name='With intervals')
else:
    course_name = medals.loc[0, 'Course']
    medals.to_excel('medalList_{}.xlsx'.format(course_name), index = False)
    with pd.ExcelWriter('super_list{}.xlsx'.format(course_name)) as writer:
        registers.to_excel(writer, sheet_name='Full')
        user_det.to_excel(writer, sheet_name='With intervals')

flag = input('Generate graphics? y/n: ')

if(flag == 'y'):

    des_coll = list(filter(lambda k: '-DISC' in k, list(registers)))
    des_coll.append('mobileHW')
    des_coll.append('mobileOS')

    flagRunners = input('Include not-finishers? y/n: ')
    if(flagRunners == 'y'):
        detection = registers[registers['racerStatus'] != 'PRESTART'].reset_index(drop = True)
        detection = detection[detection['racerStatus'] != 'GIVEUP'].loc[:, des_coll].reset_index(drop = True)
    else:
        detection = registers[registers['racerStatus'] == 'FINISH'].loc[:, des_coll].reset_index(drop = True)

    for ind5, r5 in detection.iterrows():
        if('iPhone' in detection.loc[ind5, 'mobileHW']):
            detection.loc[ind5, 'mobileHW'] = 'iOS'
            detection.loc[ind5, 'mobileOS'] = 'iOS ' + detection.loc[ind5, 'mobileOS'].split('.')[0]
        else:
            detection.loc[ind5, 'mobileHW'] = 'Android'
            detection.loc[ind5, 'mobileOS'] = detection.loc[ind5, 'mobileOS'].split('.')[0]

    #registers.to_excel('super_list{}.xlsx'.format(course_name), index = False)
    #medals.to_excel('medalList_{}.xlsx'.format(course_name), index = False)

    new_col = [x.split('-')[0] for x in detection.columns.to_list()]
    detection.columns = new_col

    detection_overall = detection.copy()
    detection_overall = detection_overall.iloc[:,:-2]
    wr = input('Reguralize?: y/n ')
    detection_ios = detection.copy()
    detection_android = detection.copy()
    detection_ios = detection_ios[detection_ios['mobileHW'] == 'iOS']
    detection_ios = detection_ios.iloc[:,:-2]
    detection_android = detection_android[detection_android['mobileHW'] == 'Android']
    detection_android = detection_android.iloc[:,:-2]
    if(wr == 'y'):
        detection_overall = detection_overall.loc[:, (detection_overall != 0).any(axis=0)]
        detection_ios = detection_ios.loc[:, (detection_ios != 0).any(axis=0)]
        detection_android = detection_android.loc[:, (detection_android != 0).any(axis=0)]


    detection_mean = detection_overall.sum(axis=0).sum() / detection_overall.shape[1]
    detection_std = detection_overall.sum(axis=0).std()
    poor_st = []
    poor_st_rate = []

    if(detection_overall.shape[1] > 80):
        fSize = (30, 22)
    else:
        fSize = (22, 14)



    fig, ax1 = plt.subplots( figsize=fSize, dpi = 300)
    ax1.set_xlabel("Beacons", fontsize=18)
    ax1.set_ylabel("# of detections",  fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.set_title('Detection rate of {} course'.format(course_name), fontsize=22, fontweight='bold')


    p1 = ax1.bar(detection_overall.columns, detection_overall.sum(axis=0), label = 'Number of detection')
    extra_info = ax1.scatter([],[], label = "Perfomace: {}%".format(round(detection_mean/detection_overall.shape[0]*100,1)), color = 'white')
    avg = ax1.axhline(y=detection_mean, color = 'red', lw = 1.5, linestyle='--', label = 'Average detection rate: {}/{}'.format(round(detection_mean), detection_overall.shape[0]))
    lines = [p1,avg, extra_info]
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15)

    for lbl, txt in enumerate(detection_overall.sum(axis=0)):
        if(txt<(detection_mean - detection_std)):
            ax1.annotate(txt, (detection_overall.columns[lbl], detection_overall.sum(axis=0)[lbl]), size=14,ha='center')
            poor_st.append(detection_overall.columns[lbl])
            poor_st_rate.append(detection_overall.sum(axis=0)[lbl])

    plt.figtext(0.05, -0.1, "Author: Bazarbay Alisher"
                        + "\nCourse Name: {}".format(course_name)
                        + "\nIntuition: By average, each ST was detected {} times OR {}%".format(round(detection_mean), round(detection_mean/detection.shape[0]*100,1))
                        #+"\nST with poor detection rate: {}".format(poor_st)##,
                        ,fontsize=20, wrap=True)
    fig.tight_layout()
    plt.xticks(rotation=90)
    fig.savefig("{}_detectionRate.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')

    #poorBeac = input('List of poor Beacons? y/n ')
    #if(poorBeac == 'y'):
    #    dfPoor = pd.DataFrame(list(zip(poor_st, poor_st_rate)), columns =['ST', 'Rate'])
    #    dfPoor = pd.to_excel('poorBeacons.xlxs')


    detection_mean_ios = detection_ios.sum(axis=0).sum() / detection_ios.shape[1]
    detection_std_ios = detection_ios.sum(axis=0).std()

    detection_mean_android = detection_android.sum(axis=0).sum() / detection_android.shape[1]
    detection_std_android = detection_android.sum(axis=0).std()
    fig_os, axs = plt.subplots(2, figsize=(20, 18), dpi = 300)
    fig_os.suptitle('Beacon detection rate of {} course by OS'.format(course_name), fontsize=22, fontweight='bold', y = 0.95)
    axs[0].set_title('iOS', fontsize=20)
    axs[1].set_title('Android', fontsize=20)
    colormap = plt.cm.gist_ncar

    #rate_ios = axs[0].scatter([], [], marker = ' ',label = 'Detection: {}'.format(round( detection_mean_ios,1)))
    ios = axs[0].bar(detection_ios.columns, detection_ios.sum(axis=0), label = 'iOS', alpha = 0.5, color = colormap(0.12))
    extra_ios = axs[0].scatter([],[], label = "Perfomace: {}%".format(round(detection_mean_ios/detection_ios.shape[0]*100,1)), color = 'white')
    avg_ios = axs[0].axhline(y=detection_mean_ios, color = 'black', lw = 1.5, linestyle='--', label = 'Average detection rate: {}/{} '.format(round(detection_mean_ios, 1), detection_ios.shape[0]))
    #for lbl1, txt1 in enumerate(detection_ios.sum(axis=0)):
    #    if(txt1<(detection_mean_ios - detection_std_ios)):
    #        axs[0].annotate(txt1, (detection_ios.columns[lbl1], detection_ios.sum(axis=0)[lbl1] + 0.5), size=12,ha='center')
    lines1 = [ios,avg_ios,extra_ios]
    axs[0].legend(lines1, [l.get_label() for l in lines1], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15)


    android = axs[1].bar(detection_android.columns, detection_android.sum(axis=0), label = 'Android', alpha = 0.5, color = colormap(0.7))
    extra_android = axs[0].scatter([],[], label = "Perfomace: {}%".format(round(detection_mean_android/detection_android.shape[0]*100,1)), color = 'white')
    avg_android = axs[1].axhline(y=detection_mean_android, color = 'black', lw = 1.5, linestyle='--', label = 'Average detection rate: {}/{}'.format(round(detection_mean_android, 1), detection_android.shape[0]))
    #for lbl2, txt2 in enumerate(detection_android.sum(axis=0)):
    #    if(txt2<(detection_mean_android - detection_std_android)):
    #        axs[1].annotate(txt2, (detection_android.columns[lbl2], detection_android.sum(axis=0)[lbl2]+0.5), size=12,ha='center')
    lines2 = [android,avg_android,extra_android]
    axs[1].legend(lines2, [l.get_label() for l in lines2], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15);



    axs[0].set_xticklabels(detection_ios.columns, rotation=90);
    axs[1].set_xticklabels(detection_android.columns, rotation=90);
    fig_os.savefig("{}_detectionRate_OS.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')

    os_ver = sorted(list(set(detection['mobileOS'].to_list())))
    dictOS = {}
    volume = []
    for ver in os_ver:
        dftemp = detection.copy()
        dftemp = dftemp[dftemp['mobileOS'] == ver]
        dftemp = dftemp.iloc[:,:-2]
        meanOS = dftemp.sum(axis=0).sum() / dftemp.shape[1]
        rate = round(meanOS/dftemp.shape[0]*100,3)
        dictOS[ver] = rate
        volume.append(dftemp.shape[0])

    keys = list(dictOS.keys());
    values = list(dictOS.values());

    figOS = plt.figure(figsize = (15,10), dpi=300)
    axOS = figOS.add_subplot(111)
    axOS1 = axOS.twiny()
    axOS1.set_xticks([])

    axOS.set_title('{} | detection perfomance by OS versions'.format(course_name), fontsize=15, fontweight='bold')

    for x,y,ss in zip(keys,values,volume):
        if('iOS' in x):
            axOS.scatter(x,y, s = ss*300, alpha=0.5, color = colormap(0.12));
        else:
            axOS.scatter(x,y, s = ss*300, alpha=0.5, color = colormap(0.7));

        axOS.scatter([],[], label = '{} rate: {}'.format(x,round(y,1)), color = 'white');
        axOS1.scatter([],[], label = '{} detection: {}/{}'.format(x,round(detection_overall.shape[1]*y/100), detection_overall.shape[1]), color = 'white');


    axOS.set_ylim(min(values)-5, max(values)+5)

    for size in range(0,len(dictOS)):
        axOS.annotate(volume[size], (keys[size], values[size]), size=12,ha='center', va = 'center')


    axOS.legend(fontsize='medium', bbox_to_anchor=(1, 1.009), loc='upper left');
    axOS1.legend( fontsize='medium', bbox_to_anchor=(1, 0.8), loc='upper left');

    figOS.savefig("{}_rateOSversion.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')


    df_runners = registers.copy()
    df_runners=df_runners[df_runners['racerStatus'] == 'FINISH']
    df_runners = df_runners.sort_values(by=['cutoffTime']).loc[:, des_coll].reset_index(drop = True)
    df_runners.columns = new_col
    df_runners = df_runners.iloc[:,:-2]
    df_top = df_runners.iloc[:int(df_runners.shape[0]*0.2), :]
    df_bottom = df_runners.iloc[-int(df_runners.shape[0]*0.2):, :]

    if(wr == 'y'):
        df_top = df_top.loc[:, (df_top != 0).any(axis=0)]
        df_bottom = df_bottom.loc[:, (df_bottom != 0).any(axis=0)]

    top_mean = df_top.sum(axis=0).sum() / df_top.shape[1]
    std_top = df_top.sum(axis=0).std()

    bottom_mean = df_bottom.sum(axis=0).sum() / df_bottom.shape[1]
    bottom_std = df_bottom.sum(axis=0).std()

    fig_port, axPort = plt.subplots(2, figsize=(20, 24), dpi = 400)
    axPort[0].set_title('Top 20% by Finish Time', fontsize=20, fontweight='bold')
    axPort[1].set_title('Bottom 20% by Finish Time', fontsize=20, fontweight='bold')
    axPort[0].tick_params(axis='both', which='major', labelsize=12)
    axPort[1].tick_params(axis='both', which='major', labelsize=12)

    #rate_ios = axs[0].scatter([], [], marker = ' ',label = 'Detection: {}'.format(round( detection_mean_ios,1)))
    top = axPort[0].bar(df_top.columns, df_top.sum(axis=0), label = 'Top 20% racers')
    extra_top = axPort[0].scatter([],[], label = "Perfomace: {}%".format(round(top_mean/df_top.shape[0]*100,1)), color='white')
    avg_top = axPort[0].axhline(y=top_mean, color = 'red', lw = 1.5, linestyle='--', label = 'Average detection rate: {}/{} '.format(round(top_mean, 1), df_top.shape[0]))

    lines_top = [top,avg_top,extra_top]
    axPort[0].legend(lines_top, [l.get_label() for l in lines_top], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15)


    bottom = axPort[1].bar(df_bottom.columns, df_bottom.sum(axis=0),color = 'C1', label = 'Bottom 20% racers')
    extra_bottom = axPort[0].scatter([],[], label = "Perfomace: {}%".format(round(bottom_mean/df_bottom.shape[0]*100,1)), color='white')
    avg_bottom = axPort[1].axhline(y=bottom_mean, color = 'red', lw = 1.5, linestyle='--', label = 'Average detection rate: {}/{}'.format(round(bottom_mean, 1), df_bottom.shape[0]))

    lines_bottom = [bottom,avg_bottom,extra_bottom]
    axPort[1].legend(lines_bottom, [l.get_label() for l in lines_bottom], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15);



    axPort[0].set_xticklabels(df_top.columns, rotation=90);
    axPort[1].set_xticklabels(df_bottom.columns, rotation=90);
    fig_port.savefig("{}_rateTopBottom.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')



    top_mean = df_top.sum(axis=0).sum() / df_top.shape[1]
    std_top = df_top.sum(axis=0).std()

    bottom_mean = df_bottom.sum(axis=0).sum() / df_bottom.shape[1]
    bottom_std = df_bottom.sum(axis=0).std()

    fig_portS, axPort = plt.subplots(figsize=(15, 10), dpi = 300)
    axPort.set_title('{} | perfomance by top/bottom 20% runners'.format(course_name), fontsize=15, fontweight='bold')


    #rate_ios = axs[0].scatter([], [], marker = ' ',label = 'Detection: {}'.format(round( detection_mean_ios,1)))
    top = axPort.plot(df_top.columns, df_top.sum(axis=0), label = 'Top 20% by Finish Time')
    bottom = axPort.plot(df_bottom.columns, df_bottom.sum(axis=0), label = 'Bottom 20% Finish Time')

    extra_top = axPort.scatter([],[], label = "Top runners: {}%".format(round(top_mean/df_top.shape[0]*100,1)))
    extra_bottom = axPort.scatter([],[], label = "Bottom runners: {}%".format(round(bottom_mean/df_bottom.shape[0]*100,1)))


    lines_topbot = [extra_top,extra_bottom]
    axPort.legend(lines_topbot, [l.get_label() for l in lines_topbot], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=15);



    axPort.set_xticklabels(df_top.columns, rotation=90);
    fig_portS.savefig("{}_rateOSversionScatter.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')



    df_brands = registers.copy()
    df_brands = df_brands[df_brands['racerStatus'] == 'FINISH'].loc[:, des_coll].reset_index(drop = True)

    for brandind, brandrow in df_brands.iterrows():
        temp = df_brands.loc[brandind, 'mobileHW']
        if('iPhone' in temp):
            df_brands.loc[brandind, 'mobileHW'] = 'iPhone'
        elif('-' in temp):
            if('SM-' in temp): df_brands.loc[brandind, 'mobileHW'] = 'Samsung'
            elif( (len(temp.split('-')[0]) == 3) & (len(temp.split('-')[1]) == 3) ):
                df_brands.loc[brandind, 'mobileHW'] = 'Huawei'
            else:
                df_brands.loc[brandind, 'mobileHW'] = 'Others'
        elif(('MI' in temp) | ('Mi' in temp) | ('mi' in temp)):
            df_brands.loc[brandind, 'mobileHW'] = 'Xiaomi'
        else:
            df_brands.loc[brandind, 'mobileHW'] = 'Others'


    hw_ver = sorted(list(set(df_brands['mobileHW'].to_list())))
    dictHW = {}
    volumeHW = []
    for hw in hw_ver:
        dftemp = df_brands.copy()
        dftemp = dftemp[dftemp['mobileHW'] == hw]
        dftemp = dftemp.iloc[:,:-2]
        meanHW = dftemp.sum(axis=0).sum() / dftemp.shape[1]
        rate = round(meanHW/dftemp.shape[0]*100,3)
        dictHW[hw] = rate
        volumeHW.append(dftemp.shape[0])

    keys = list(dictHW.keys());
    values = list(dictHW.values());

    figHW = plt.figure(figsize = (12,10), dpi=300)
    axHW = figHW.add_subplot(111)
    axHW1 = axHW.twiny()
    axHW1.set_xticks([])
    axHW.set_title('{} | detection perfomance by brands'.format(course_name), fontsize=15, fontweight='bold')

    for x,y,ss in zip(keys,values,volumeHW):
        axHW.scatter(x,y, s = ss*300, alpha=0.5);
        axHW.scatter([],[], label = '{} rate: {}'.format(x,round(y,1)), color = 'white');
        axHW1.scatter([],[], label = '{} detection: {}/{}'.format(x,round(detection_overall.shape[1]*y/100), detection_overall.shape[1]), color = 'white');


    axHW.set_ylim(min(values)-5, max(values)+5)

    for size in range(0,len(dictHW)):
        axHW.annotate(volumeHW[size], (keys[size], values[size]), size=12,ha='center', va = 'center')

    axHW.legend(fontsize='medium', bbox_to_anchor=(1, 1.008), loc='upper left');
    axHW1.legend( fontsize='medium', bbox_to_anchor=(1, 0.8), loc='upper left');

    figHW.savefig("{}_rateBrand.pdf".format(course_name), format = 'pdf', dpi=300, bbox_inches='tight')


    fixCol, fixDet =  detection_overall.columns, detection_overall.sum(axis=0)
    fixDic = {'Beacon':fixCol, '#Detections': fixDet}
    fixDf = pd.DataFrame(fixDic)
    fixDf.to_excel('Beacons.xlsx')
