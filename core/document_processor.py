import logging
import PyPDF2 # Using PyPDF2 as it's listed in requirements.txt implicitly
from typing import List
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessorError(Exception):
    """Custom exception for DocumentProcessor errors."""
    pass

class DocumentProcessor:
    """
    A class responsible for processing documents, including PDF text extraction
    and text splitting into manageable chunks.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initializes the DocumentProcessor.

        Args:
            chunk_size (int, optional): The target size for text chunks (in characters).
                                        Defaults to DEFAULT_CHUNK_SIZE from settings.
            chunk_overlap (int, optional): The overlap between consecutive chunks (in characters).
                                           Defaults to DEFAULT_CHUNK_OVERLAP from settings.
        """
        self.chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size and self.chunk_size > 0 : # check chunk_size > 0 to avoid division by zero if chunk_size is used as step
            logger.warning(f"Chunk overlap ({self.chunk_overlap}) is greater than or equal to chunk size ({self.chunk_size}). This may lead to redundant or empty chunks. Setting overlap to chunk_size / 2 or 0 if chunk_size is small.")
            if self.chunk_size > 10: # Arbitrary threshold
                 self.chunk_overlap = self.chunk_size // 2
            else:
                 self.chunk_overlap = 0


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts all text content from a given PDF file.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            str: The concatenated text content from all pages of the PDF.

        Raises:
            DocumentProcessorError: If the PDF file cannot be opened or processed.
            FileNotFoundError: If the pdf_path does not exist.
        """
        logger.info(f"Attempting to extract text from PDF: {pdf_path}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF '{pdf_path}' has {num_pages} pages.")
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() or "" # Add "or """ to handle None from extract_text
            logger.info(f"Successfully extracted text from PDF: {pdf_path}. Total characters: {len(text)}")
            return text
        except FileNotFoundError:
            logger.error(f"PDF file not found at path: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF file {pdf_path}: {e}")
            raise DocumentProcessorError(f"Failed to extract text from PDF {pdf_path}: {e}")

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Splits a long text into smaller chunks with a specified overlap.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            logger.warning("split_text_into_chunks called with empty text.")
            return []

        if self.chunk_size <= 0:
            logger.warning(f"chunk_size ({self.chunk_size}) is not positive. Returning the whole text as one chunk.")
            return [text]

        logger.info(f"Splitting text of length {len(text)} into chunks of size {self.chunk_size} with overlap {self.chunk_overlap}.")

        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunks.append(text[start_index:end_index])
            start_index += self.chunk_size - self.chunk_overlap
            if start_index >= len(text) and len(text) > self.chunk_size : # Avoid infinite loop if overlap makes step 0 or negative and text is small
                 break
            if self.chunk_size - self.chunk_overlap <= 0 and len(text) > self.chunk_size : # handle cases where step is 0 or less
                logger.warning("Chunk step is non-positive due to overlap >= size. Breaking split to avoid infinite loop.")
                break


        # Remove empty or whitespace-only chunks that might result from splitting
        chunks = [chunk for chunk in chunks if chunk.strip()]
        logger.info(f"Text split into {len(chunks)} non-empty chunks.")
        return chunks

if __name__ == '__main__':
    # This is an example of how to use the DocumentProcessor.
    # It requires a sample PDF file for testing extraction.
    print("DocumentProcessor Example")

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Example 1: Text Splitting
    sample_text = "This is a long sample text that needs to be split into several chunks for processing. Each chunk should be manageable. The overlap helps maintain context between chunks."
    print(f"\nOriginal text: '{sample_text}'")
    text_chunks = processor.split_text_into_chunks(sample_text)
    print("Text chunks:")
    for i, chunk in enumerate(text_chunks):
        print(f"  Chunk {i+1}: '{chunk}' (Length: {len(chunk)})")

    # Example 2: PDF Text Extraction (requires a PDF file)
    # Create a dummy PDF for testing if one doesn't exist
    # Note: PyPDF2 cannot create PDFs, only read/manipulate.
    # For a real test, place a sample.pdf in the script's directory or provide a path.
    sample_pdf_path = "sample.pdf"

    # Check if a dummy PDF can be created or if a placeholder is needed
    # For now, we will just show how to call it, assuming a PDF exists.
    print(f"\nAttempting to extract text from '{sample_pdf_path}' (if it exists).")
    print("Please place a file named 'sample.pdf' in the same directory as this script to test PDF extraction.")

    try:
        # Create a dummy PDF file for testing (this part requires reportlab or similar)
        # Since we don't have reportlab as a dependency, we'll skip actual creation
        # and rely on the user providing a sample.pdf.
        # For demonstration, let's assume a file exists or handle the FileNotFoundError.

        # if not os.path.exists(sample_pdf_path):
        #     print(f"'{sample_pdf_path}' not found. PDF extraction test will be skipped unless you create it.")
        # else:
        #    pdf_text = processor.extract_text_from_pdf(sample_pdf_path)
        #    print(f"\nExtracted PDF text (first 500 chars):\n'{pdf_text[:500]}...'")

        #    print("\nSplitting extracted PDF text into chunks:")
        #    pdf_chunks = processor.split_text_into_chunks(pdf_text)
        #    print(f"PDF text split into {len(pdf_chunks)} chunks.")
        #    if pdf_chunks:
        #        print(f"  First PDF chunk: '{pdf_chunks[0][:100]}...'")
        pass # Actual call is commented out

    except FileNotFoundError:
        print(f"File '{sample_pdf_path}' not found. Skipping PDF extraction part of the example.")
    except DocumentProcessorError as e:
        print(f"Error processing PDF: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing: {e}")

    print("\nDocumentProcessor example finished.")
