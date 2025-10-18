# 📁 Data Ingestion Pipeline
import json
from pathlib import Path
from typing import List, Dict, Any

class LegalDocumentIngester:
    """
    Handles ingestion of legal documents from various sources
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
    def create_sample_legal_documents(self):
        """
        Create sample legal documents for demonstration purposes
        In production, this would fetch from actual legal sources
        """
        sample_docs = {
            "san_francisco_rent_control": {
                "title": "San Francisco Rent Control Ordinance",
                "jurisdiction": "San Francisco, CA",
                "content": """
                SAN FRANCISCO RENT CONTROL ORDINANCE
                
                Section 37.3: Rent Increase Limitations
                (a) No landlord may increase rent by more than 7% per year, or the percentage increase 
                in the Consumer Price Index, whichever is less.
                
                (b) Rent increases are limited to once per 12-month period.
                
                (c) Landlords must provide 30 days written notice for rent increases.
                
                Section 37.9: Eviction Protections
                (a) Landlords may only evict tenants for just cause as defined in this ordinance.
                
                (b) Just cause includes: non-payment of rent, violation of lease terms, 
                owner move-in, demolition, or substantial rehabilitation.
                
                (c) Tenants have the right to contest evictions in court.
                """,
                "source": "San Francisco Rent Board",
                "last_updated": "2024-01-15"
            },
            
            "california_civil_code": {
                "title": "California Civil Code - Tenant Rights",
                "jurisdiction": "California",
                "content": """
                CALIFORNIA CIVIL CODE - TENANT RIGHTS
                
                Section 1941: Landlord's Duty to Maintain Habitable Premises
                (a) A landlord must maintain the premises in a habitable condition.
                
                (b) Habitable conditions include: effective waterproofing, plumbing, heating, 
                electrical systems, clean and sanitary buildings, adequate trash receptacles.
                
                Section 1942: Tenant's Right to Repair and Deduct
                (a) If a landlord fails to repair conditions that materially affect health and safety,
                a tenant may repair the condition and deduct the cost from rent.
                
                (b) The repair cost cannot exceed one month's rent.
                
                Section 1946: Notice Requirements for Termination
                (a) For month-to-month tenancies, either party may terminate with 30 days notice.
                
                (b) For tenancies of one year or more, 60 days notice is required.
                """,
                "source": "California Civil Code",
                "last_updated": "2024-01-10"
            },
            
            "fair_housing_act": {
                "title": "Fair Housing Act - Protected Classes",
                "jurisdiction": "Federal",
                "content": """
                FAIR HOUSING ACT - PROTECTED CLASSES
                
                Section 804: Prohibited Discrimination
                It is unlawful to discriminate against any person in the terms, conditions, 
                or privileges of sale or rental of a dwelling because of:
                
                (a) Race or color
                (b) Religion
                (c) National origin
                (d) Sex (including gender identity and sexual orientation)
                (e) Familial status (presence of children under 18)
                (f) Disability
                
                Section 805: Discriminatory Practices
                Prohibited practices include:
                - Refusing to rent or sell housing
                - Setting different terms or conditions
                - Providing different services or facilities
                - Making discriminatory statements or advertisements
                """,
                "source": "U.S. Department of Housing and Urban Development",
                "last_updated": "2024-01-05"
            },
            
            "austin_tenant_rights": {
                "title": "Austin Tenant Rights Guide",
                "jurisdiction": "Austin, TX",
                "content": """
                AUSTIN TENANT RIGHTS GUIDE
                
                Eviction Notice Requirements:
                (a) For non-payment of rent: 3-day notice to pay or quit
                (b) For lease violations: 3-day notice to cure or quit
                (c) For no-cause termination: 30-day notice for month-to-month tenancies
                
                Security Deposit Rights:
                (a) Landlords must return security deposits within 30 days of move-out
                (b) Itemized deductions must be provided in writing
                (c) Tenants may dispute deductions in small claims court
                
                Repair Rights:
                (a) Tenants may request repairs in writing
                (b) Landlords have reasonable time to make repairs
                (c) Tenants may withhold rent for uninhabitable conditions (with proper notice)
                """,
                "source": "Austin Tenants Council",
                "last_updated": "2024-01-12"
            }
        }
        
        # Save sample documents
        for doc_id, doc_data in sample_docs.items():
            doc_path = self.raw_dir / f"{doc_id}.json"
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created {len(sample_docs)} sample legal documents")
        return sample_docs
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all legal documents from the raw directory (JSON files only)
        """
        documents = []
        for doc_path in self.raw_dir.glob("*.json"):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    documents.append(doc_data)
            except Exception as e:
                print(f"⚠️ Error loading JSON file {doc_path}: {e}")
        
        print(f"📚 Loaded {len(documents)} legal documents")
        return documents

# Initialize the ingester
ingester = LegalDocumentIngester()

# Create sample documents
sample_docs = ingester.create_sample_legal_documents()

# Load JSON documents
documents = ingester.load_documents()
 