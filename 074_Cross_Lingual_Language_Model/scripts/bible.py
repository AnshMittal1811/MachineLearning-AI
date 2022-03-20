# Usage : python bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages --books $books --cell_error $cell_error
"""
csv_path : folder containing the csvs folder
output : folder in which the objective folders will be created (mono or para)
data_type (optional, default = "para") : monolingual one ("mono") or parallel one ("para")
languages (optional, default = langues_nt) : list of languages to be considered in alphabetical order and separated by a comma : e.g. 'Bafia,Bulu,Ewondo'. (these languages must be included in the list of languages)
books (optional, defautlt = livres_all) : list of the books of the bibles to be considered separated by a comma (there must exist for each of these books a books.csv file in csv_path/csvs/)
cell_error (optional, defautlt = "__Error__") : text to be used to mark erroneous text pairs during webscrapping (these pairs are excluded from the data)    
old_only : use only old testament
new_only : use only new testament
"""

import argparse
import os
import csv

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
        

abreviation = {
    "Francais":"fr", "Anglais":"en",
    #"ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", 
    #"ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"
    "BIBALDA_TA_PELDETTA":"MASS", "Bulu":"Bulu", "Guiziga":"Guiz", "Fulfulde_Adamaoua":"Fulf",  
    "Fulfulde_DC":"Fulf", "KALATA_KO_SC_Gbaya":"Gbay", "KALATA_KO_DC_Gbaya":"Gbay", 
    "Kapsiki_DC":"Kaps", "Tupurri":"Tupu",
    ##############
    "Bafia":"Bafi", "Dii":"Dii", "Ejagham":"Ejag", "Ghomala":"Ghom", "Vute":"Vute", "Limbum":"Limb", 
    "MKPAMAN_AMVOE_Ewondo":"Ewon", "Mofa":"Mofa", "Mofu_Gudur":"Mofu", "Ngiemboon":"Ngie", 
    "Doyayo":"Doya", "Guidar":"Guid", "Peere_Nt&Psalms":"Peer", "Samba_Leko":"Samb", 
    "Du_na_sdik_na_wiini_Alaw":"Du_n"
}

livres_ot = [
    # Old Testament
    'GEN.', 'EXO.', 'LEV.', 'NUM.', 'DEU.', 'JOS.', 'JDG.', 'RUT.',   '1SA.', '2SA.', '1KI.', '2KI.', 
    '1CH.', '2CH.', 'EZR.', 'NEH.', 'EST.', 'JOB.', 'PSA.', 'PRO.', 'ECC.', 'SNG.',  'ISA.', 'JER.', 
    'LAM.', 'EZK.', 'DAN.', 'HOS.', 'JOL.', 'AMO.', 'OBA.', 'JON.', 'MIC.', 'NAM.', 'HAB.', 'ZEP.', 
    'HAG.', 'ZEC.', 'MAL.'
]

livres_nt = [
    # New Testament
    'MAT.', 'MRK.', 'LUK.', 'JHN.', 'ACT.', 'ROM.', '1CO.', '2CO.', 'GAL.', 'EPH.', 'PHP.', 'COL.', 
    '1TH.', '2TH.', '1TI.', '2TI.', 'TIT.', 'PHM.', 'HEB.', 'JAS.', '1PE.',  '2PE.', '1JN.', '2JN.', 
    '3JN.', 'JUD.', 'REV.'
]

livres_all = livres_ot + livres_nt

# languages present in the old testament 
langues_at = [
    'Francais', "Anglais",  "BIBALDA_TA_PELDETTA", 'Bulu',  'Guiziga', "Fulfulde_Adamaoua",  
    "Fulfulde_DC", 'KALATA_KO_SC_Gbaya', 'KALATA_KO_DC_Gbaya', 'Kapsiki_DC', 'Tupurri'
]

# languages present in the new testament 
langues_nt = [
    # languages present in the old and new testament 
    'Francais', "Anglais",  "BIBALDA_TA_PELDETTA", 'Bulu',  'Guiziga', "Fulfulde_Adamaoua",  
    "Fulfulde_DC", 'KALATA_KO_SC_Gbaya', 'KALATA_KO_DC_Gbaya', 'Kapsiki_DC', 'Tupurri',
           
    # languages present only in the new testament
    'Bafia', 'Dii', 'Ejagham', 'Ghomala', 'Vute', 'Limbum', 'MKPAMAN_AMVOE_Ewondo', 'Mofa', 
    "Mofu_Gudur", "Ngiemboon", 'Doyayo', "Guidar", 'Peere_Nt&Psalms', 'Samba_Leko', 
    "Du_na_sdik_na_wiini_Alaw"
]

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Africa bible data")

    # main parameters
    parser.add_argument("--csv_path", type=str, default="", help="folder containing the csvs folder")
    parser.add_argument("--output_dir", type=str, default="", help="folder in which the objective folders will be created (mono or para)")
    parser.add_argument("--data_type", type=str, default="para", help="monolingual (mono) one or parallel one (para)")
    parser.add_argument("--languages", type=str, default="", help="list of languages to be considered in alphabetical order and separated by a comma : e.g. 'Bafia,Bulu,Ewondo'. (these languages must be included in the list of languages above)")
    parser.add_argument("--books", type=str, default="", help="list of the books of the bibles to be considered separated by a comma (there must exist for each of these books a books.csv file in ../csvs/)")
    parser.add_argument("--cell_error", type=str, default="__Error__", help="text to be used to mark erroneous text pairs during webscrapping (these pairs are excluded from the data).")
    parser.add_argument("--old_only", type=bool_flag, default=False, help="use only old testament")
    parser.add_argument("--new_only", type=bool_flag, default=False, help="use only new testament")
    
    return parser

def get_abreviation(lang_name):
    return abreviation.get(lang_name, lang_name)
    
process = []
def get_data_from_bible(csv_path, output_dir, data_type = "para", langues=[], livres=[], cell_error = "__Error__"):
    """
    csv_path : folder containing the csvs folder
    output : folder in which the objective folders will be created (mono or para)
    data_type (optional, default = "para") : monolingual one ("mono") or parallel one ("para")
    langues (optional, default = langues_nt) : list of languages to be considered (these languages must be included in the list of languages above)
    livres (optional, defautlt = livres_all) : list of the books of the bibles to be considered (there must exist for each of these books a livre.csv file in ../csvs/)
    cell_error (optional, defautlt = "__Error__") : text to be used to mark erroneous text pairs during webscrapping (these pairs are excluded from the data)    
    """
    global process 
    
    # If no language is specified, all languages are selected.
    if langues == [] :
        langues = langues_nt
    langues = sorted(langues)

    # If no book is specified, all books are selected.
    if livres == [] :
        livres = livres_all
        
    l = len(langues)
    for i in range(l-1):
        for j in range(i+1, l):
            li = langues[i]
            lj = langues[j]

            samples = 0
            errors = 0

            li_abrev_temp = get_abreviation(li)
            lj_abrev_temp = get_abreviation(lj)
            abrev = sorted([li_abrev_temp, lj_abrev_temp])
            li_abrev = abrev[0]
            lj_abrev = abrev[1]
            #repertoire = output_dir+"/"+data_type
            repertoire = output_dir
            if not os.path.exists(repertoire):
                os.makedirs(repertoire)

            if data_type == "para":
                repertoire = repertoire +"/"+ li_abrev +'-'+ lj_abrev
                if not os.path.exists(repertoire):
                    os.makedirs(repertoire)
                repertoire = [repertoire + "/" + li_abrev + '-' + lj_abrev + "." for _ in range(2)]
            elif data_type == "mono":
                #repertoire = [repertoire + "/" + li_abrev, repertoire +"/"+ lj_abrev]
                repertoire = [repertoire, repertoire]
                for rep in list(set(repertoire)) :
                    if not os.path.exists(rep):
                        os.makedirs(rep)
                repertoire = [r +"/" for r in repertoire]
                
            if li_abrev_temp == li_abrev:
                li_abrev, lj_abrev = repertoire[0] + li_abrev + '.txt', repertoire[1] + lj_abrev + '.txt'
            else :
                lj_abrev, li_abrev = repertoire[0] + li_abrev + '.txt', repertoire[1] + lj_abrev + '.txt'
                
            with open(li_abrev, 'w') as txtfile1:
                with open(lj_abrev, 'w') as txtfile2:
            
                    for fichier in livres:
                        try :
                            with open(csv_path+"/csvs/"+fichier+'csv', 'r') as csvfile:
                                f_csv = csv.reader(csvfile)
                                filenames = []
                                try :
                                    fieldnames = next(f_csv)
                                except StopIteration :
                                    pass
                                try :
                                    index_i = fieldnames.index(li)
                                    index_j = fieldnames.index(lj)
                                    #versert = fieldnames.index("livre.chapitre.verset")
                                    for ligne in f_csv:
                                        x_i = ligne[index_i] 
                                        y_i =  ligne[index_j]
                                        if x_i != cell_error and y_i != cell_error :
                                            samples = samples + 1
                                            txtfile1.writelines(x_i+'\n')
                                            txtfile2.writelines(y_i+'\n')
                                        else :
                                            errors = errors + 1
                                except :
                                    pass
                        except :
                            pass
            
            stat = False
            if li != lj :
                if not li+"-"+lj in process :
                    stat = True
                    print(li,"-",lj)
                    process.append(li+"-"+lj)
            else :
                if not li in process :
                    stat = True
                    print(li)
                    process.append(li)
                 
            if stat :    
                print("======= Read "+str(samples+errors)+" totals samples")
                print("======= Delete "+str(errors)+" samples") 
                print("======= Save "+str(samples)+" samples")  

            # If one of the two languages is in the old testament for both languages and the other is not,
            # build the mono data for the one in the two separations
            built_mono = {li : False, lj : False}
            ai = li in langues_at
            aj = lj in langues_at
            if ai and not aj :
                built_mono[li] = True
            elif not ai and aj :
                built_mono[lj] = True
            for langue in [li, lj] :
                if built_mono[langue] :
                    get_data_from_bible(csv_path, output_dir, data_type = "mono", langues = [langue, langue], livres = livres, cell_error = cell_error)


def main(params):
    get_data_from_bible(params.csv_path, params.output_dir, params.data_type, params.languages, params.books, params.cell_error)
    
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    # check parameters
    assert os.path.isdir(params.csv_path), "csv path not found"
    assert params.data_type in ["para", "mono"], "Invalid data type"
    assert not params.old_only or not params.new_only

    if params.books.replace(" ", "") != "" :
        params.books = params.books.split(",")
        assert all([book in livres_all for book in params.books]), "Invalid books"
    else :
        params.books = []
    if params.old_only :
        params.books = livres_ot
    if params.new_only :
        params.books = livres_nt
        
    if params.languages.replace(" ", "") != "" :
        params.languages = params.languages.split(",")
        assert all([(language in langues_nt) for language in params.languages]), "Invalid languages"
    else :
        params.languages = []

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    # run experiment
    main(params)