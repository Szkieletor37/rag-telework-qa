from logging import DEBUG, getLogger

from google import genai

from config import CLEANED_MD_DIR, GEMINI_API_KEY_FREE, PROMPT_DIR, QUESTION_DIR, setup_environment


setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def generate_questions() -> None:
    """カテゴリごとに質問を生成する関数（カテゴリあたりの生成数はbaseプロンプト内に記載）"""

    categories = ["enumerate", "conditional", "entity", "multi_hop", "unanswerable", "wording_variant"]

    model = "gemini-3.6-flash"

    client = genai.Client(api_key=GEMINI_API_KEY_FREE)

    base_prompt = (PROMPT_DIR / "question_gen_base.md").read_text(encoding="utf-8")

    target_md = (CLEANED_MD_DIR / "telework_guide.md").read_text(encoding="utf-8")

    system_prompt = f"""
{base_prompt}

# 対象資料
{target_md}

"""

    for category in categories[:1]:
        logger.info(f"category: {category} の質問生成を開始します...")

        category_prompt = (PROMPT_DIR / "category" / f"{category}.md").read_text(encoding="utf-8")

        full_prompt = f"""
{system_prompt}

# 個別カテゴリー指示
{category_prompt}
"""
        
        # token数のカウント
        # 事前予測
        total_tokens = client.models.count_tokens(
            model=model,
            contents=full_prompt
        )
        logger.info(f"  total入力tokens: {total_tokens.total_tokens}")

        # 質問生成

        interaction = client.interactions.create(
            model=model,
            input=full_prompt,
        )
        logger.info(f"  質問生成に成功しました。")
        logger.info(f"  使用したtoken数: {interaction.usage}")

        # カテゴリごとにyamlファイルに保存
        output_file = QUESTION_DIR / f"{category}.yaml"
        output_file.write_text(interaction.output_text, encoding="utf-8")
        logger.info(f"  生成した質問を, {output_file} に保存しました。")


if __name__ == "__main__":
    generate_questions()