from logging import DEBUG, basicConfig, getLogger

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def main():
    logger.info("Hello from rag-telework-qa!")

if __name__ == "__main__":
    # logging のフォーマット
    FORMATTER = '[%(levelname)s %(asctime)s] - %(message)s'
    # basicConfig は何も設定されてないときのみ有効
    basicConfig(format=FORMATTER)

    main()
