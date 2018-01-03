# with open('./data/MT/records-sc.txt') as f:
#     lines = f.read().splitlines()
#
# new_lines = []
# max_year = 0
# min_year = 2018
#
# for line in lines:
#     new_line = ['sentiment classification']
#     info = line.split('\t')
#     info[0] = info[0].lower().strip()
#     info[2] = str(int(info[2]) - 2013)
#     info[1] = info[1].lower().replace('-', '').replace('abstract', '').replace(':', '').replace('"', '').strip()
#     new_line += info
#     new_lines.append('\t'.join([new_line[0], info[0], info[2], info[3]]))
#
#
# with open('./data/MT/topic-title-SC.txt', 'w') as f:
#     f.write('\n'.join(new_lines))
# print(new_lines)

# with open('./data/MT/title-abstract-SC.txt') as f:
#     lines = f.read().splitlines()
#
# news = []
# for line in lines:
#     info = line.split('\t')
#     info[3] = str(float(info[3]) / (5 - int(info[2])))
#     news.append('\t'.join(info))
#
# with open('./data/MT/title-abstract-SC.txt', 'w') as f:
#     f.write('\n'.join(news))
new_lines = []
with open('data/MT/title-abstract-MT.txt') as f:
    lines = f.read().splitlines()
    for line in lines:
        info = line.split('\t')
        info.append('0')
        new_line = []
        abs = info[1]
        count = 0
        for i, sentence in enumerate(abs.split('.')):
            if count > 4:
                break
            if len(sentence.split(' ')) > 10:
                info[1] = sentence
                info[-1] = str(count)
                new_lines.append('\t'.join(info))
                count += 1
print(new_lines)
with open('./data/MT/title-abstract-mt-pos.txt', 'w') as f:
    f.write('\n'.join(new_lines))