"""
Pytest configuration and fixtures for DocumentIndex tests.
"""

import pytest
import asyncio
from typing import Generator

# Sample document content for testing
SAMPLE_10K_TEXT = """FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)
OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, 2024

Commission File Number: 001-12345

ACME CORPORATION
(Exact name of registrant as specified in its charter)

Delaware                                    12-3456789
(State of incorporation)                    (I.R.S. Employer Identification No.)

123 Main Street, New York, NY 10001
(Address of principal executive offices)

Trading Symbol: ACME

CIK: 0001234567

PART I

ITEM 1. BUSINESS

Overview

ACME Corporation ("ACME," "the Company," "we," or "us") is a leading provider of 
innovative technology solutions. Founded in 1995, we have grown to become a global 
leader in our industry.

Our Revenue for fiscal year 2024 was $15.2 billion, representing a 12% increase 
from the prior year. Net Income was $2.3 billion, with Diluted EPS of $4.52.

For more details on our financial performance, see Item 7. Management's Discussion 
and Analysis and Note 15 in the financial statements.

Products and Services

We offer a comprehensive portfolio of products and services across three main segments:

1. Enterprise Solutions
2. Consumer Products  
3. Cloud Services

See Appendix G for a complete list of our product offerings.

ITEM 1A. RISK FACTORS

Investing in our common stock involves risks. You should carefully consider the 
following risk factors, as well as the other information in this report.

Climate Change Risks

Climate change may adversely affect our operations and supply chain. Extreme weather 
events could disrupt our manufacturing facilities and distribution networks. We have 
implemented environmental sustainability initiatives to mitigate these risks.

Regulatory Compliance

We are subject to various laws and regulations in the jurisdictions where we operate.
Changes in regulatory requirements could increase our compliance costs. See Item 3
for information about legal proceedings.

Cybersecurity Risks

Our business relies heavily on information technology systems. A significant 
cybersecurity breach could harm our reputation and financial results.

ITEM 2. PROPERTIES

Our corporate headquarters is located in New York, NY. We also maintain regional 
offices in San Francisco, London, and Singapore.

ITEM 3. LEGAL PROCEEDINGS

We are involved in various legal proceedings arising in the ordinary course of 
business. For details, see Note 12 to the financial statements.

PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS 
OF OPERATIONS

Executive Summary

Fiscal 2024 was a year of strong performance for ACME Corporation. We achieved 
record revenue and continued to invest in research and development.

Revenue Analysis

Total revenue increased 12% to $15.2 billion:
- Enterprise Solutions: $8.5 billion (+15%)
- Consumer Products: $4.2 billion (+8%)
- Cloud Services: $2.5 billion (+18%)

Gross Margin was 45.9%, compared to 44.2% in the prior year.

Operating Expenses

Research and development expenses were $2.1 billion (13.8% of revenue).
Selling, general and administrative expenses were $1.8 billion (11.8% of revenue).

Liquidity and Capital Resources

We ended the year with $5.3 billion in cash and cash equivalents. Total Assets 
were $28.5 billion. Shareholders' Equity was $18.2 billion.

ITEM 8. FINANCIAL STATEMENTS

The consolidated financial statements are included in Appendix A.

NOTE 12. COMMITMENTS AND CONTINGENCIES

We have operating lease commitments totaling $450 million. Legal contingencies 
are described in Item 3.

NOTE 15. SEGMENT INFORMATION

We operate in three reportable segments as described in Item 1.
"""

SAMPLE_EARNINGS_CALL_TEXT = """ACME Corporation Q4 2024 Earnings Call

Operator: Good afternoon. Welcome to ACME Corporation's Fourth Quarter 2024 
Earnings Conference Call. I would like to remind everyone that this call is 
being recorded.

At this time, I'd like to turn the call over to Jane Smith, Vice President of 
Investor Relations. Please go ahead.

Jane Smith - VP, Investor Relations:
Thank you, operator. Good afternoon, everyone, and thank you for joining us today.
With me on the call are John Doe, our Chief Executive Officer, and Sarah Johnson, 
our Chief Financial Officer.

Before we begin, I'd like to remind you that during this call we may make 
forward-looking statements regarding our future business expectations.

I'll now turn the call over to John.

John Doe - CEO:
Thank you, Jane, and good afternoon, everyone. I'm pleased to report that Q4 was 
another strong quarter for ACME. We delivered revenue of $4.1 billion, up 14% 
year-over-year, and EPS of $1.25.

Looking at our segments:
- Enterprise Solutions grew 16% to $2.3 billion
- Consumer Products grew 10% to $1.1 billion
- Cloud Services grew 22% to $700 million

I'll now turn it over to Sarah for the financial details.

Sarah Johnson - CFO:
Thank you, John. Let me walk you through our Q4 financial results in more detail.

Revenue was $4.1 billion, above our guidance of $3.9 to $4.0 billion. Gross margin 
was 46.2%, up 80 basis points year-over-year.

Operating expenses were well controlled at $950 million. We generated operating 
cash flow of $1.2 billion in the quarter.

For the full year, we achieved revenue of $15.2 billion and EPS of $4.52.

Looking ahead to Q1 2025, we expect revenue of $3.8 to $4.0 billion and EPS of 
$1.05 to $1.15.

Operator: We will now begin the question and answer session.

Analyst 1: Hi, thanks for taking my question. Can you talk about the competitive 
environment in Cloud Services?

John Doe: Sure. The cloud market remains highly competitive, but we're seeing 
strong demand for our differentiated offerings. Our AI-powered solutions have 
been particularly well received.

Analyst 2: What's your outlook for capital expenditures in 2025?

Sarah Johnson: We expect capex of approximately $1.5 billion in 2025, primarily 
for data center expansion and R&D facilities.

John Doe: Thank you all for joining us today. We look forward to speaking with 
you again next quarter.

Operator: This concludes today's conference call. You may now disconnect.
"""


@pytest.fixture
def sample_10k_text() -> str:
    """Sample 10-K document text"""
    return SAMPLE_10K_TEXT


@pytest.fixture
def sample_earnings_call_text() -> str:
    """Sample earnings call transcript"""
    return SAMPLE_EARNINGS_CALL_TEXT


@pytest.fixture
def sample_short_text() -> str:
    """Short sample text for quick tests"""
    return """FORM 10-K
    
PART I

ITEM 1. BUSINESS

We are a technology company providing software solutions.
Our revenue was $5 billion in 2024.

ITEM 1A. RISK FACTORS

Market competition is intense.
Cybersecurity threats continue to evolve.

PART II

ITEM 7. MD&A

Revenue grew 15% year over year.
"""


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
