import logging
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ReportCompilerAgentError(Exception):
    """Custom exception for ReportCompilerAgent errors."""
    pass

class ReportCompilerAgent(BaseAgent):
    """
    Agent responsible for compiling all the refined (or initially written) chapters
    into a single, coherent report string, following a given outline.
    It can also add a title, table of contents (optional), and other formatting.
    """

    def __init__(self, add_table_of_contents: bool = True):
        """
        Initializes the ReportCompilerAgent.

        Args:
            add_table_of_contents (bool): Whether to generate and add a simple
                                          table of contents (based on Markdown outline).
                                          Defaults to True.
        """
        super().__init__(agent_name="ReportCompilerAgent") # No LLM needed directly for simple compilation
        self.add_table_of_contents = add_table_of_contents
        logger.info(f"ReportCompilerAgent initialized. Add Table of Contents: {self.add_table_of_contents}")

    def _parse_markdown_outline(self, markdown_outline: str) -> List[Dict[str, any]]:
        """
        Parses a Markdown list outline into a structured list of chapters/sections.
        Each item in the list is a dict with 'level' and 'title'.
        Example:
        - Chapter 1
          - Section 1.1
        becomes:
        [ {'title': 'Chapter 1', 'level': 1}, {'title': 'Section 1.1', 'level': 2} ]
        """
        parsed_outline = []
        lines = markdown_outline.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            level = 0
            title = line

            # Determine level by leading characters ('-', '*', '+', or spaces for indentation)
            # This is a simple parser, might need to be more robust for complex Markdown.
            if line.startswith("- ") or line.startswith("* ") or line.startswith("+ "):
                level = 1
                title = line[2:].strip()
                # Check for further indentation for sub-levels (e.g. "  - Section")
                # This part is tricky with simple line parsing; a true Markdown parser would be better.
                # For now, we assume simple list structure.
                # A more robust way would be to count leading spaces for sub-items.

                # Simplified: Count leading spaces before the bullet for sub-levels
                # This requires consistent spacing in the outline.
                leading_spaces = 0
                temp_line = line
                while temp_line.startswith("  "): # Two spaces per indent level
                    level +=1
                    temp_line = temp_line[2:]

                # Re-extract title if level was adjusted by spaces
                if level > 1 :
                    # find first non-space character after initial bullet stripping attempts
                    stripped_line = line.lstrip()
                    if stripped_line.startswith("- ") or stripped_line.startswith("* ") or stripped_line.startswith("+ "):
                         title = stripped_line[2:].strip()
                    else: # If it's just indented text without a new bullet (less common for outlines)
                         title = stripped_line.strip()


            elif line.startswith("#"): # Also support Markdown headers if used in outline
                level = line.count("#")
                title = line.lstrip("# ").strip()

            else: # Assume it's a top-level item if no clear marker (could be problematic)
                  # Or, if the outline is purely list-based, non-bullet items might be ignored or handled differently
                  # For now, let's assume well-formed Markdown lists.
                  # If line doesn't start with a list marker, we might skip it or treat as level 1.
                  # For robustness, let's only process lines that seem like list items or headers.
                continue # Skip lines not recognized as outline items


            if title: # Only add if a title was successfully extracted
                parsed_outline.append({"title": title, "level": level, "content_key": title}) # Use title as key for content

        logger.debug(f"Parsed outline: {parsed_outline}")
        return parsed_outline

    def _generate_table_of_contents(self, parsed_outline: List[Dict[str, any]]) -> str:
        """Generates a Markdown table of contents from the parsed outline."""
        if not self.add_table_of_contents or not parsed_outline:
            return ""

        toc = "## 目录\n\n"
        for item in parsed_outline:
            title = item['title']
            level = item['level']
            # Create a simple anchor link (GitHub-style, needs to be URL-encoded and lowercased)
            # This is a naive anchor generation. Real Markdown processors have more complex rules.
            anchor = title.lower().replace(" ", "-").translate(str.maketrans("", "", '()[]{}<>/"\'?!,:;.')) # Basic sanitization

            indent = "  " * (level - 1)
            toc += f"{indent}- [{title}](#{anchor})\n"
        toc += "\n---\n" # Separator
        return toc

    def run(self,
            report_title: str,
            markdown_outline: str,
            chapter_contents: Dict[str, str],
            report_topic_details: Optional[Dict[str, any]] = None
           ) -> str:
        """
        Compiles the report.

        Args:
            report_title (str): The main title of the report.
            markdown_outline (str): The Markdown formatted outline of the report.
            chapter_contents (Dict[str, str]): A dictionary where keys are chapter titles
                                               (matching those in the outline) and values
                                               are the text content for each chapter.
            report_topic_details (Optional[Dict[str, any]]): The output from TopicAnalyzerAgent,
                                                             containing generalized topics and keywords.
                                                             Used for an optional introduction/summary.

        Returns:
            str: The fully compiled report as a single Markdown string.

        Raises:
            ReportCompilerAgentError: If essential inputs are missing or formatting fails.
        """
        self._log_input(report_title=report_title, outline_length=len(markdown_outline),
                        num_chapters_content=len(chapter_contents), report_topic_details_present=bool(report_topic_details))

        if not report_title:
            raise ReportCompilerAgentError("Report title cannot be empty.")
        if not markdown_outline:
            raise ReportCompilerAgentError("Markdown outline cannot be empty.")
        if not chapter_contents:
            logger.warning("Compiling report with no chapter contents provided.")
            # Depending on strictness, could raise error or return minimal report.

        final_report_parts = []

        # 1. Report Title
        final_report_parts.append(f"# {report_title}\n")

        # (Optional) Add a brief introduction/summary based on topic details
        if report_topic_details:
            topic_cn = report_topic_details.get("generalized_topic_cn", "未提供")
            keywords_cn_list = report_topic_details.get("keywords_cn", [])
            keywords_cn_str = ", ".join(keywords_cn_list) if keywords_cn_list else "无"

            intro_summary = f"## 引言\n\n本报告围绕主题“**{topic_cn}**”展开，"
            if keywords_cn_list:
                intro_summary += f"重点探讨与关键词“{keywords_cn_str}”相关的议题。\n"
            intro_summary += "报告旨在提供对此主题的深入分析和全面概述。\n\n---\n"
            final_report_parts.append(intro_summary)


        # 2. Parse Outline and Generate Table of Contents (Optional)
        # The parsed_outline will determine the order and hierarchy of chapters.
        # The keys in chapter_contents should match the 'title' from parsed_outline.
        parsed_outline = self._parse_markdown_outline(markdown_outline)
        if not parsed_outline:
            logger.warning("Could not parse the provided Markdown outline. Report structure might be incorrect.")
            # Fallback: try to iterate chapter_contents if outline parsing fails?
            # For now, we rely on a parsable outline.

        if self.add_table_of_contents:
            toc_md = self._generate_table_of_contents(parsed_outline)
            if toc_md:
                final_report_parts.append(toc_md)

        # 3. Add Chapters based on Parsed Outline
        # We use the parsed_outline to ensure correct order and hierarchy.
        # Chapter titles from the outline are used as keys to fetch content from chapter_contents.

        # To create proper Markdown headers for chapters, we need to know their level.
        # The _parse_markdown_outline should provide this.

        for item in parsed_outline:
            chapter_key = item['title'] # The key used in chapter_contents should match this.
            level = item['level']

            content = chapter_contents.get(chapter_key)

            if content is None:
                logger.warning(f"No content found for chapter/section: '{chapter_key}'. It will be omitted or marked as TBD.")
                # Optionally add a placeholder
                # final_report_parts.append(f"\n{'#' * level} {chapter_key}\n\n*内容待定 (Content TBD)*\n")
                continue # Skip if no content

            # Add chapter title as a Markdown header
            # Ensure there's a newline before starting a new chapter's content
            # Also, add an anchor for the TOC. This is a simplified anchor.
            anchor = chapter_key.lower().replace(" ", "-").translate(str.maketrans("", "", '()[]{}<>/"\'?!,:;.'))
            final_report_parts.append(f"\n<a id=\"{anchor}\"></a>\n{'#' * level} {chapter_key}\n\n{content}\n")

        # Fallback for any content provided in chapter_contents but not found in outline
        # This might indicate an issue with outline parsing or content keys.
        # For now, we prioritize the outline. Content not in outline is ignored.
        # outlined_keys = {item['title'] for item in parsed_outline}
        # for key, content in chapter_contents.items():
        #     if key not in outlined_keys:
        #         logger.warning(f"Content for '{key}' was provided but not found in the parsed outline. It will be appended at the end or ignored.")
        #         # final_report_parts.append(f"\n## {key} (未在目录中)\n\n{content}\n") # Example of appending

        compiled_report = "".join(final_report_parts)
        self._log_output(compiled_report)
        return compiled_report.strip()


if __name__ == '__main__':
    print("ReportCompilerAgent Example")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        compiler_agent_with_toc = ReportCompilerAgent(add_table_of_contents=True)
        compiler_agent_no_toc = ReportCompilerAgent(add_table_of_contents=False)

        report_title_ex = "关于ABMS系统的综合分析报告"

        # Example topic details from TopicAnalyzerAgent
        topic_details_ex = {
            "generalized_topic_cn": "先进战斗管理系统（ABMS）及其影响",
            "generalized_topic_en": "Advanced Battle Management System (ABMS) and its Implications",
            "keywords_cn": ["ABMS", "JADC2", "军事技术", "指挥控制"],
            "keywords_en": ["ABMS", "JADC2", "Military Technology", "C2"]
        }

        # Example Markdown outline (could come from OutlineGeneratorAgent)
        outline_ex = """
        - 章节一：ABMS系统概述
          - 1.1 ABMS定义与目标
          - 1.2 发展背景
        - 章节二：核心技术分析
          - 2.1 数据与网络
          - 2.2 人工智能应用
        - 章节三：挑战与展望
        - 章节四：结论
        """
        # A more complex outline for testing parser
        outline_complex_ex = """
        # 第一章：引言 (H1)
        ## 1.1 研究背景 (H2)
        - 1.1.1 技术驱动 (List item, effective level 3 if under H2)
        ## 1.2 研究意义 (H2)
        # 第二章：系统分析 (H1)
        - 2.1 架构设计 (List item, effective level 2)
          - 2.1.1 模块A (List item, effective level 3)
          - 2.1.2 模块B (List item, effective level 3)
        - 2.2 功能特性 (List item, effective level 2)
        # 第三章：总结 (H1)
        """


        # Example chapter contents (keys should match titles in the outline)
        # For simplicity, using the simple outline_ex titles.
        chapters_data_ex = {
            "章节一：ABMS系统概述": "这是关于ABMS系统概述的详细内容...",
            "1.1 ABMS定义与目标": "ABMS旨在实现全域信息的无缝共享和协同决策...",
            "1.2 发展背景": "随着现代战争形态的演变，对高效指挥控制系统的需求日益迫切...",
            "章节二：核心技术分析": "ABMS的核心技术涵盖了多个方面...",
            "2.1 数据与网络": "强大的数据链和弹性网络是ABMS的基础...",
            "2.2 人工智能应用": "AI在ABMS中用于辅助决策、目标识别等...",
            "章节三：挑战与展望": "ABMS面临技术成熟度、成本控制和安全等多重挑战。未来，ABMS有望...",
            "章节四：结论": "综上所述，ABMS是未来智能化军队建设的关键组成部分...",
            "额外章节不在大纲中": "这部分内容不应出现在基于大纲的报告中，除非有特殊处理。"
        }

        # Test with TOC
        print("\n--- Compiling report WITH Table of Contents ---")
        compiled_report_with_toc = compiler_agent_with_toc.run(
            report_title=report_title_ex,
            markdown_outline=outline_ex,
            chapter_contents=chapters_data_ex,
            report_topic_details=topic_details_ex
        )
        print(f"\n**Compiled Report (with TOC):**\n{compiled_report_with_toc}")

        # Test without TOC
        print("\n\n--- Compiling report WITHOUT Table of Contents ---")
        compiled_report_no_toc = compiler_agent_no_toc.run(
            report_title=report_title_ex,
            markdown_outline=outline_ex,
            chapter_contents=chapters_data_ex,
            report_topic_details=None # No intro summary
        )
        print(f"\n**Compiled Report (no TOC, no intro):**\n{compiled_report_no_toc}")

        # Test complex outline parsing (just the parsing part)
        print("\n\n--- Testing Complex Outline Parsing ---")
        parsed_complex = compiler_agent_with_toc._parse_markdown_outline(outline_complex_ex)
        print("Parsed Complex Outline Structure:")
        for item in parsed_complex:
            print(item)
        # And generate TOC for it
        # toc_complex = compiler_agent_with_toc._generate_table_of_contents(parsed_complex)
        # print(f"\nGenerated TOC for Complex Outline:\n{toc_complex}")


    except ReportCompilerAgentError as e:
        print(f"Agent error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nReportCompilerAgent example finished.")
