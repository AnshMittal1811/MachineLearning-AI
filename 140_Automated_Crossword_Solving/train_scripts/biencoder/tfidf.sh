BASENAME=$1
JSONL_NAME="${BASENAME}/docs.jsonl"
DB_NAME="${BASENAME}/docs.db"
FOLDER_NAME="${BASENAME}/tfidf/"

rm ${DB_NAME}
python3 DrQA/scripts/retriever/build_db.py ${JSONL_NAME} ${DB_NAME}
python3 DrQA/scripts/retriever/build_tfidf.py ${DB_NAME} ${FOLDER_NAME} --tokenizer simple