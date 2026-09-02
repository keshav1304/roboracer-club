import { MapPin, Clock } from "lucide-react";
import PageHero from "@/components/layout/PageHero";

const events = [
    {
        id: 1,
        title: "Intro to ROS2 Workshop",
        date: "Feb 15, 2026",
        time: "6:00 PM - 8:00 PM",
        location: "Levine Hall, Room 307",
        description: "Learn the basics of Robot Operating System 2, the standard for autonomous systems.",
        category: "Workshop"
    },
    {
        id: 2,
        title: "Spring Kickoff Meeting",
        date: "Jan 28, 2026",
        time: "7:00 PM - 9:00 PM",
        location: "Towne Building, Room 100",
        description: "Join us for our first meeting of the semester! Pizza will be provided.",
        category: "General"
    },
    {
        id: 3,
        title: "RoboRacer-mini Time Trials",
        date: "Mar 10, 2026",
        time: "10:00 AM - 4:00 PM",
        location: "Penn Engineering Quad",
        description: "First round of time trials for the semester. Come test your algorithms!",
        category: "Competition"
    }
];

export default function CalendarPage() {
    return (
        <div className="flex flex-col">
            <PageHero
                title="Calendar"
                subtitle="Upcoming workshops, meetings, and races."
            />

            <div className="bg-gradient-to-b from-brand-dark to-brand-[#ec4899]">
                <section className="py-16 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-5xl mx-auto">
                        <div className="space-y-6">
                            {events.map((event) => (
                                <div key={event.id} className="bg-white/5 rounded-2xl p-6 border border-white/10 hover:border-brand-magenta/50 transition-colors duration-300 flex flex-col md:flex-row gap-6">
                                    <div className="flex-shrink-0 flex flex-col items-center justify-center bg-brand-dark rounded-xl p-4 w-full md:w-32 border border-white/5">
                                        <span className="text-sm text-brand-blue font-bold uppercase">{event.date.split(' ')[0]}</span>
                                        <span className="text-3xl font-bold text-white">{event.date.split(' ')[1].replace(',', '')}</span>
                                        <span className="text-xs text-gray-500">{event.date.split(' ')[2]}</span>
                                    </div>

                                    <div className="flex-grow">
                                        <div className="flex flex-wrap gap-2 mb-2">
                                            <span className="text-xs font-medium px-2 py-1 rounded-full bg-white/10 text-gray-300">
                                                {event.category}
                                            </span>
                                        </div>
                                        <h3 className="text-xl font-bold text-white mb-2">{event.title}</h3>
                                        <p className="text-gray-400 mb-4">{event.description}</p>

                                        <div className="flex flex-col sm:flex-row gap-4 text-sm text-gray-500">
                                            <div className="flex items-center">
                                                <Clock size={16} className="mr-2 text-brand-magenta" />
                                                {event.time}
                                            </div>
                                            <div className="flex items-center">
                                                <MapPin size={16} className="mr-2 text-brand-blue" />
                                                {event.location}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
