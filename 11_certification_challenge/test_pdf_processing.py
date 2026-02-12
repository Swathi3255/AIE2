# 🧪 PDF Processing Test Script
"""
Test script to demonstrate PDF processing capabilities
Run this script to test PDF document ingestion
"""

from src.data_ingestion import LegalDocumentIngester

def test_pdf_processing():
    """
    Test PDF processing functionality
    """
    print("🧪 Testing PDF Processing Capabilities")
    print("=" * 50)
    
    # Initialize ingester
    ingester = LegalDocumentIngester()
    
    # Get document summary
    summary = ingester.get_document_summary()
    print("\n📊 Document Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if summary["pdf_documents"] > 0:
        print(f"\n📄 Found {summary['pdf_documents']} PDF documents:")
        for pdf_file in summary["pdf_files"]:
            print(f"  - {pdf_file}")
        
        # Process PDF documents
        print("\n🔄 Processing PDF documents...")
        pdf_documents = ingester.process_pdf_documents()
        
        # Display sample content from first PDF
        if pdf_documents:
            first_pdf = pdf_documents[0]
            print(f"\n📋 Sample PDF Document:")
            print(f"  Title: {first_pdf['title']}")
            print(f"  Jurisdiction: {first_pdf['jurisdiction']}")
            print(f"  Source: {first_pdf['source']}")
            print(f"  Page Count: {first_pdf.get('page_count', 'Unknown')}")
            print(f"  Content Preview: {first_pdf['content'][:200]}...")
    else:
        print("\n📄 No PDF documents found in data/raw/ directory")
        print("   To test PDF processing, add some PDF files to data/raw/")
    
    # Load all documents
    print("\n📚 Loading all documents...")
    all_documents = ingester.load_documents()
    
    print(f"\n✅ Successfully loaded {len(all_documents)} total documents")
    
    return all_documents

if __name__ == "__main__":
    test_pdf_processing()
