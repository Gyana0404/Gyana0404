"""Regenerate the synthetic fake/real text samples for offline software testing."""
from pathlib import Path
import random
import pandas as pd
ROOT=Path(__file__).resolve().parent
rng=random.Random(20261104)
real_subjects=["the transport department","the city council","the university research office","the meteorological agency","the public health department","the central bank","the election commission","the energy regulator","the district administration","the statistical office"]
real_actions=["published its monthly report","released an update","announced the completion of a review","presented the annual figures","issued a public notice","shared the preliminary findings","confirmed the scheduled inspection","opened the application window"]
real_topics=["road maintenance","rainfall monitoring","school enrolment","electricity demand","consumer prices","public transport","water quality","employment data","budget spending","vaccination services","crop conditions","air quality"]
real_details=["The document includes tables and the method used to compile the figures","Officials said the information will be reviewed again next quarter","The notice lists the locations and the dates covered by the update","The agency said the full statement is available through its public information office","The report compares the latest figures with the previous reporting period","The department said the measure follows its published procedure"]
fake_openers=["SHOCKING claim","Viral post","Unverified message","Sensational online claim","A widely shared rumour","BREAKING claim"]
fake_topics=["a secret nationwide scheme","a hidden payment plan","an instant cure","a dramatic overnight change","a secret election instruction","an unbelievable market announcement","a concealed government order","a miracle treatment"]
fake_claims=["users are urged to forward the message immediately","the post provides no verifiable document or named source","the claim relies on anonymous screenshots","the message says the announcement is being hidden from the public","the post promises a guaranteed result without supporting evidence","the source asks readers to ignore established reporting"]
fake_endings=["No official record is linked in the message","The text gives no date, location or primary source","The claim is presented without independently checkable evidence","Readers are told to share it before it is removed","The post does not identify a responsible organization"]
rows=[]
for _ in range(500):
    text=f"On {rng.choice(['Monday','Tuesday','Wednesday','Thursday','Friday'])}, {rng.choice(real_subjects)} {rng.choice(real_actions)} about {rng.choice(real_topics)}. {rng.choice(real_details)}. The update is part of its regular public information process."
    rows.append({"text":text,"label":"REAL","record_origin":"synthetic_demo"})
for _ in range(500):
    text=f"{rng.choice(fake_openers)}: {rng.choice(fake_topics)} is happening right now, according to a post circulating online. {rng.choice(fake_claims)}. {rng.choice(fake_endings)}."
    rows.append({"text":text,"label":"FAKE","record_origin":"synthetic_demo"})
rng.shuffle(rows)
pd.DataFrame(rows).to_csv(ROOT/"data"/"fake_news_dataset.csv",index=False)
print("Generated 1,000 synthetic demo rows.")
