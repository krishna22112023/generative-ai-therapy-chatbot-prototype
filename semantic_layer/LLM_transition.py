from semantic_router import RouteLayer
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from conf import config

class LLMTransition:
    def get_routes(self):
        Counsellor = Route(
            name="Counsellor",
            utterances = [
            "What health screenings are recommended for young adults in Singapore?",
            "How can I maintain my fitness with a busy schedule?",
            "What are the recommended physical activity guidelines for adults?",
            "How to manage stress in daily life?",
            "What vaccinations are recommended for adults in Singapore?",
            "How to prevent heat-related illnesses in Singapore's climate?",
            "What should I expect during a routine health check-up?",
            "How to maintain a balanced diet on a budget?",
            "What are common workplace injuries and how can I prevent them?",
            "How to improve sleep quality and manage sleep deprivation?",
            "What mental health support services are available in Singapore?",
            "How to stay hydrated in hot and humid weather?",
            "What are the signs of heat exhaustion?",
            "How to maintain good foot health if I stand a lot at work?",
            "What should I do if I get sick and need to take time off work?",
            "How to manage chronic health conditions while working full-time?",
            "What health insurance options are available in Singapore?",
            "How to prevent and treat blisters from physical activities?",
            "What is the process for reporting workplace injuries in Singapore?",
            "How to maintain good hygiene in shared living or working spaces?",
            "What are the nutritional guidelines for a healthy adult diet?",
            "How to manage allergies in Singapore's environment?",
            "What should I know about sun protection in tropical climates?",
            "How to prevent and treat insect bites in Singapore?",
            "What are the signs of overtraining or excessive exercise?",
            "How to maintain good oral hygiene?",
            "What should I know about vision care and eye health?",
            "How to prevent back pain from sedentary work?",
            "What are the health considerations for shift workers?",
            "How to manage asthma in Singapore's climate?",
            "What should I know about mental health assessments?",
            "How to maintain a healthy weight with a sedentary job?",
            "What are the health risks of sleep disorders?",
            "How to prevent and treat common skin infections?",
            "What should I know about hearing protection in noisy environments?",
            "How to manage motion sickness during travel?",
            "What are the health considerations for overseas travel?",
            "How to prevent and treat sports injuries?",
            "What should I know about health screening packages in Singapore?",
            "How to maintain cardiovascular fitness?",
            "What are the health implications of body modifications like piercings and tattoos?",
            "How to manage digestive issues with a busy lifestyle?",
            "What should I know about smoking cessation programs in Singapore?",
            "How to prevent and treat common running injuries like shin splints?",
            "What are the health considerations for different occupations?",
            "How to manage stress-related skin conditions?",
            "What should I know about physiotherapy services in Singapore?",
            "How to maintain good posture at work?",
            "What are the health considerations for night shift workers?",
            "How to manage allergic reactions to common substances in Singapore?",
            "Can you share A Dozen Videos for Teaching Mindfulness to Teens?",
            "Can you share tips for Stress Management?",
            "Why Sleep is So Important"
        ]
        )
        Survey = Route(
            name="Survey",
            utterances=[
                "I want to take the survey now",
                "take survey now",
                "Can I take the survey now?",
                "I would like to take the survey now"
            ]
        )
        RAG_NS = Route(
            name="RAG_NS",
            utterances=[
                # Pre-NS Utterances
                "When is my enlistment date?",
                "How do I prepare for NS?",
                "What should I bring on enlistment day?",
                "How long is BMT?",
                "What's the pre-enlistment IPPT?",
                "Can I choose my vocation?",
                "What's the medical screening process like?",
                "How fit do I need to be before NS?",
                "What documents do I need for enlistment?",
                "Can I defer my NS?",
                "What happens during the first week of NS?",
                "How to cut hair for NS?",
                "What's PES status?",
                "NS packing list",
                "How to mentally prepare for NS?",
                "vocation selection process"
                # During-NS Utterances
                "BMT schedule",
                "NS leave policy",
                "How to apply for off days?",
                "What's cookhouse food like?",
                "NS allowance details",
                "How to book out?",
                "Vocational training duration",
                "IPPT requirements during NS",
                "Can I use phone in camp?",
                "NS promotion criteria",
                "How to report sick in NS?",
                "Camp facilities",
                "NS physical training routine",
                "How to maintain civilian relationships during NS?",
                "NS uniform care instructions",
                # Post-NS Utterances
                "Reservist schedule",
                "IPPT for NSmen",
                "How often are ICT callups?",
                "NS benefits after ORD",
                "Overseas travel notification for NSmen",
                "How long do reservist obligations last?",
                "NSmen IPPT monetary award",
                "Can I skip ICT?",
                "NSmen job search assistance",
                "How to update NS contact details?",
                "NS55 credits usage",
                "MR (Mindef Reserve) status requirements",
                "NSmen fitness routines",
                "Reservist pay rates",
                "NS learning portal access after ORD",
                # General NS Utterances
                "NS ranks explained",
                "NS acronyms list",
                "SAF core values",
                "NS medical benefits",
                "NS recognition awards",
                "NS for permanent residents",
                "NS history in Singapore",
                "NS and university admission",
                "NS and career impact",
                "NS skills transferable to civilian life"
            ]
        )
        routes = [Counsellor,Survey,RAG_NS]
        router = RouteLayer(encoder=OpenAIEncoder(), routes=routes)
        return router


def main():
    router = LLMTransition().get_routes()
    print(router("hello"))

if __name__ == "__main__":
    main()