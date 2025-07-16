from uagents import Model
#VH stands for Vaccine Hesitancy

class VHRegionRequest(Model):
    region: str
    
class VHRegionResponse(Model):
    response: str
    status: str

class VHAgeBandRequest(Model):
    age_band: str

class VHAgeBandResponse(Model):
    response: str
    status: str

class VHAgeGroupRequest(Model):
    age_group: str

class VHAgeGroupResponse(Model):
    response: str
    status: str 

class VHSexRequest(Model):
    sex: str

class VHSexResponse(Model):
    response: str
    status: str

class VHEthnicityRequest(Model):
    ethnicity: str

class VHEthnicityResponse(Model):
    response: str
    status: str

class VHReligionRequest(Model):
    religion: str
    
class VHReligionResponse(Model):
    response: str
    status: str

class VHDisabilityRequest(Model):
    status: str  # disabled|non-disabled

class VHDisabilityResponse(Model):
    response: str
    status: str

class VHUnderlyingHealthRequest(Model):
    status: str  # has_condition|none

class VHUnderlyingHealthResponse(Model):
    response: str
    status: str

class VHHealthGeneralRequest(Model):
    condition: str  # has_condition|none

class VHHealthGeneralResponse(Model):
    response: str
    status: str

class VHCEVRequest(Model):
    status: str  # cev|non_cev

class VHCEVResponse(Model):
    response: str
    status: str

class VHIMDRequest(Model):
    quintile: str  # 1-most_deprived|2|3|4|5-least

class VHIMDResponse(Model):
    response: str
    status: str

class VHEmploymentRequest(Model):
    status: str  # employed|unemployed|retired|other

class VHEmploymentResponse(Model):
    response: str
    status: str

class VHExpenseRequest(Model):
    ability: str  # can_afford|cannot_afford|other

class VHExpenseResponse(Model):
    response: str
    status: str

class VHHouseholdRequest(Model):
    type: str  # single|couple|family|other

class VHHouseholdResponse(Model):
    response: str
    status: str

class VHCaregiverRequest(Model):
    status: str  # caregiver|non_caregiver

class VHCaregiverResponse(Model):
    response: str
    status: str

class VHAgeSexRequest(Model):
    group: str  # e.g. 'All adults', '16 to 29', '30 to 49', 'Aged 50 and over'
    subgroup: str = None  # e.g. 'Men', 'Women', or None for 'All adults'
    measure: str = None  # e.g. 'Have received a vaccine (one or two doses)', etc.

class VHAgeSexResponse(Model):
    response: str
    status: str

class VHTrendsRequest(Model):
    period: str = None  # e.g. '10 December to 10 January5'
    block: str = None  # e.g. 'Vaccine offers and uptake'
    measure: str = None  # e.g. 'Have received one dose of a vaccine'
    value_type: str = None  # '%', 'LCL', 'UCL'

class VHTrendsResponse(Model):
    response: str
    status: str

class VHBarriersRequest(Model):
    block: str = None  # e.g. 'Among those who have received at least one dose of a vaccine'
    group: str = None  # e.g. 'All persons', 'Disabled', etc.
    measure: str = None  # e.g. 'Difficulty travelling to receive the vaccine'
    value_type: str = None  # '%', 'LCL', 'UCL'

class VHBarriersResponse(Model):
    response: str
    status: str

class VHReasonsRequest(Model):
    period: str = None  # e.g. 'Great Britain, 23 June to 18 July 2021'
    block: str = None   # e.g. 'Health', 'Fertility', etc.
    measure: str = None # e.g. 'I am worried about the side effects', etc.
    group: str = None   # e.g. 'All persons'

class VHReasonsResponse(Model):
    response: str
    status: str

class VHAgentRequest(Model):
    query: str

class VHAgentResponse(Model):
    response: str
    status: str


    


