"""Build Dataset/current/scam_curated.csv.

Two provenance classes, recorded in the 'source' column:
- verbatim_* : real scam messages quoted in news/scam-report articles
  (Which?, PUNCH Nigeria, ScamCheck, Detroit Free Press, MEN, lilachbullock)
- synthetic_*: postings written to match documented scam archetypes
  (task scams, advance-fee, pig-butchering gigs, impersonation).
  Synthetic rows are for TRAINING augmentation only — evaluation uses
  verbatim rows to stay honest.
"""

import csv
import os

COLS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "fraudulent", "source",
]

VERBATIM = [
    # --- PUNCH Nigeria, Nov 2023 (job-scam SMS/WhatsApp messages) ---
    ("You have passed the audit and receive a salary of 19800 naira per day. Contact information: (link).",
     "verbatim_punch_ng_2023"),
    ("I am a project manager, we are hiring a team. You can work from home, daily salary: 80000. To accept this job please click (link).",
     "verbatim_punch_ng_2023"),
    ("Hi dear, we are looking for workers. You can work from home and earn 5000-60000 naira per day. To accept this job please click (link).",
     "verbatim_punch_ng_2023"),
    ("My name is Lisa. My team is in desperate need of staff. Daily salary is 30,000NGN. If interested please contact WhatsApp (link).",
     "verbatim_punch_ng_2023"),
    ("After our initial screening, you meet our recruitment requirements, daily wages 50,000-60,000, for more details (link).",
     "verbatim_punch_ng_2023"),
    ("Hi, my name is Adella, I'm a Senior Representative Officer at Casvo Digital Services Limited. Our company is recruiting part-time/full-time online employees. Our job is very simple and does not take much time. We are offering a job opportunity that can be done in your comfort zone. You will be rewarded with N2,000 for completing a trial task.",
     "verbatim_punch_ng_2023"),
    # --- ScamCheck India examples ---
    ("Hello, I'm HR Manager at Amazon India. We have a Work From Home vacancy. Salary: ₹45,000/month. No experience needed. Registration fee: ₹999 only. Interested? Reply YES.",
     "verbatim_scamcheck_in"),
    ("Congratulations! Your profile on Naukri.com has been shortlisted for a Data Entry position at TCS. Package: ₹35,000/month. Pay ₹1,500 security deposit to confirm your slot. Contact: +91-XXXXXXXXXX",
     "verbatim_scamcheck_in"),
    ("Part-time job opportunity — Earn ₹800-₹1,200 per task by completing simple product ratings on Amazon. Join our WhatsApp group to get started. No experience needed!",
     "verbatim_scamcheck_in"),
    # --- Which? UK ---
    ("Hello! We are a regular online recruitment game company, there is no time limit and location limit, the work is very simple, you can do it at home, and the manager teaches you to make money and do things. Salary £50-300, please add WhatsApp to receive jobs.",
     "verbatim_which_uk"),
    # --- Manchester Evening News / TikTok task scam ---
    ("Hello! I'm Amelie from TikTok's HR department. We're pleased to offer you a part-time-work-from-home role to support our business expansion. Working hours; 60 to 90 minutes per day. Daily pay: 300 GPB to 600GBP. Guaranteed monthly salary: 10,000 GBP. Add me on WhatsApp to receive an 8 GBP bonus!",
     "verbatim_men_uk"),
    # --- lilachbullock / Meesho work-from-home scam ---
    ("Official Meesho Work From Home Job, part time or full time, earn ₹15,000 to ₹45,000 per month, product listing and order confirmation, no experience needed, registration fee ₹499 only (fully refundable). Interested? WhatsApp me for details.",
     "verbatim_meesho_wfh"),
    # --- Detroit Free Press 'Mia from Indeed' scam (partial verbatim) ---
    ("Your background and resume have caught the attention of several online recruiters, so we would like to offer you a job that you can do from home in your free time. This job is easy and has no time limit. Daily pay ranges from $200 to $1,000 and is paid on the same day. Join us and be a part of America's booming job market and start a career you can be proud of.",
     "verbatim_freep_indeed"),
    # --- CyberGuy Amazon remote recruitment scam (composite of quoted fragments) ---
    ("Hi! This is Lauren Whitmore from the Amazon Remote Recruitment Team. Your background makes you a great fit for a stress-free, task-based remote position supporting our merchants with quick product refreshes. About 1 to 1.5 hours of work per day, only four days a week. Pay ranges from $100 to $600 daily, plus a $5,300 base for every 30 worked days, free training, 15-20 days paid time off. Only 18 positions remaining! Applicants must be at least 23 years old. Text 'More Info' to this number to apply.",
     "verbatim_cyberguy_amazon"),
]

SYNTHETIC = [
    # Task-commission scam (crypto "earning platform")
    ("Remote Task Assistant Wanted — Earn $300-$800 daily completing simple online tasks. Work from home, flexible hours, no experience needed. Training provided free. Register on our earning platform with referral code and start earning today. Withdrawals processed daily once you complete the starter tasks.",
     "synthetic_task_scam"),
    # Advance-fee equipment scam
    ("We're hiring Home-Based Electronics Testers. You will receive a testing kit shipped to your home. A refundable deposit of $150 is required to cover shipping and insurance of the equipment, which will be reimbursed with your first paycheck. Apply via WhatsApp +1-XXX-XXX-XXXX.",
     "synthetic_advance_fee"),
    # Fake-check / overpayment scam
    ("Data Entry Clerk needed urgently. We will send you a startup check for equipment purchase. Deposit the check, keep $200 as your first week salary, and wire the remaining balance to our vendor as instructed. Salary $45/hour, paid weekly.",
     "synthetic_fake_check"),
    # Parcel mule scam
    ("Parcel Sorting Associate — work from home sorting and reshipping packages. $25 per parcel processed. No experience needed, training included. Must have reliable internet and ability to receive shipments at your residence. Apply now, positions filling fast!",
     "synthetic_mule"),
    # Pig-butchering 'gig' job (brand impersonation)
    ("Hi! I'm reaching out from the Spotify Partner Recruitment team. We're looking for remote 'streaming assistants' to help artists grow their monthly listeners. The work is simple: play tracks and confirm streams from your account. Earn $150-$400 per day, paid every Friday in Bitcoin or PayPal. Message me on Telegram @spotifypartners to get started — only 20 spots left!",
     "synthetic_pig_butchering"),
    # Mystery shopper scam
    ("Mystery Shopper Wanted! Evaluate restaurants and stores near you and get paid $100-$250 per assignment. No experience necessary. First you must purchase a 'certification badge' for $49 which will be fully refunded after your first assignment. Text SHOPS to 555-0100.",
     "synthetic_mystery_shopper"),
    # Crypto payment processor scam
    ("Crypto Payment Processor — part-time remote. Process USDT transactions for our exchange platform. Earn 5% commission on every transaction, average $500-$2,000 daily. Must be comfortable with cryptocurrency wallets. DM me on WhatsApp for the onboarding link and your referral code.",
     "synthetic_crypto_processor"),
    # Typing/data entry commission scam
    ("Fast Typing Jobs — Earn money from home typing documents. We pay per word typed. No degree or experience required. After registration you will receive assignments. Payouts via PayPal every Friday. Join our Telegram channel for daily job alerts!",
     "synthetic_typing"),
    # 'HR manager' WhatsApp style
    ("Hello dear, good day! I am Mrs. Sandra, HR Manager at Zenith Logistics. We are currently recruiting part-time staff for our remote operations team. Salary is very attractive — N150,000 weekly. No interview needed, just complete a short registration form and pay a small onboarding fee of N2,500 to activate your account. Interested? Reply YES.",
     "synthetic_hr_whatsapp"),
    # Social media 'engagement' scam
    ("Social Media Engagement Officer — remote position helping brands grow on Instagram and TikTok. Like, follow, and comment on posts daily. Earn $20-$60 per hour. Must be 18+, own a smartphone. DM 'Manager Mike' on WhatsApp to begin your first paid task today!",
     "synthetic_engagement"),
    # Online tutor advance-fee
    ("We're hiring ESL Online Tutors — teach English to students overseas via Zoom. $30-$50/hour, set your own schedule. Must complete our TEFL certification course ($199, refundable after your first month) before teaching. Send your CV and PayPal email to tutor.recruit@quickmail.com",
     "synthetic_tutor_fee"),
    # 'Government contract' scam
    ("URGENT: Federal Contract Work-From-Home Opportunity. The government is hiring remote data processors for a new census verification project. Earn $3,500-$5,000 weekly. Due to high demand only 15 positions remain. You must register and pay a $99 processing fee to receive your contract number. Contact Agent Williams at +1-800-XXX-XXXX.",
     "synthetic_gov_contract"),
    # Review-farming scam (Temu/Shein style)
    ("Product Review Specialist — work from home reviewing products on major e-commerce sites. Earn $10-$25 per review, up to $800 weekly. No experience needed. You'll receive products to review or review items online. Payment daily via gift cards or PayPal. Message me on WhatsApp +44-XXXX to start!",
     "synthetic_review_farm"),
    # 'AI training' data labeling scam
    ("AI Data Labeler — annotate images and text for AI companies from home. Earn $200-$500 daily, no experience needed! After a quick WhatsApp interview you'll get access to our labeling portal. A $25 'portal access fee' is required, fully refunded with your first payout. Hurry — spots limited!",
     "synthetic_ai_label"),
    # 'Resume distribution' scam
    ("Resume Posting Service — we will distribute your resume to 500+ companies for a one-time fee of $299. You'll hear back within 48 hours guaranteed! No job hunting needed. Pay now via credit card or Bitcoin to start your job search automation.",
     "synthetic_resume_fee"),
    # 'Director of remote work' vague high-pay
    ("DIRECTOR OF VIRTUAL OPERATIONS — $8,000-$12,000 monthly. Manage a remote team of 50+ virtual assistants. Fully remote, flexible schedule. Must have reliable internet and a quiet workspace. This position is 100% commission-based with uncapped earning potential. Apply through WhatsApp only: +91-XXXXXXXXXX",
     "synthetic_vague_exec"),
    # 'Dropshipping coach' recruitment
    ("Dropshipping Business Opportunity — start your own e-commerce empire! Our mentor will train you to make $5,000-$20,000 monthly selling products online. Investment package starts at just $499 (includes course, supplier list, and ad templates). Limited seats — enroll today via WhatsApp!",
     "synthetic_dropshipping"),
    # 'Hotel/travel review' gig scam
    ("Travel & Hospitality Reviewer — get paid to review hotels and restaurants worldwide from your phone. Earn $50-$150 per review plus free stays at partner hotels. Flexible schedule, work whenever you want. Sign up free at our portal, then complete your first paid review within 24 hours!",
     "synthetic_hotel_review"),
    # 'Insurance claim processor' scam
    ("Insurance Claims Processor — remote position processing claims for a major US insurance provider. $35/hour, 1099 contractor. You'll need to purchase our proprietary claims software license for $149 (reimbursed after 30 days). Duties include data entry and document review. Apply via text to 1-800-XXX-XXXX.",
     "synthetic_insurance"),
]


def row(text, source, idx, title=""):
    return {
        "job_id": f"curated_{idx}",
        "title": title,
        "location": "",
        "department": "",
        "salary_range": "",
        "company_profile": "",
        "description": text,
        "requirements": "",
        "benefits": "",
        "telecommuting": "",
        "has_company_logo": "",
        "has_questions": "",
        "employment_type": "",
        "required_experience": "",
        "required_education": "",
        "industry": "",
        "function": "",
        "fraudulent": 1,
        "source": source,
    }


def main():
    os.makedirs("Dataset/current", exist_ok=True)
    rows = [row(t, s, i) for i, (t, s) in enumerate(VERBATIM + SYNTHETIC)]
    with open("Dataset/current/scam_curated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    n_verbatim = sum(1 for _, s in VERBATIM if s.startswith("verbatim"))
    print(f"Wrote {len(rows)} scam postings "
          f"({n_verbatim} verbatim, {len(SYNTHETIC)} synthetic) "
          "-> Dataset/current/scam_curated.csv")


if __name__ == "__main__":
    main()
