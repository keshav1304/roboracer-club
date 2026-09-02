import Image from "next/image";
import { ArrowRight } from "lucide-react";

export default function HowItWorks() {
    return (
        <section className="py-14 border-b border-white/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center mb-12">
                    <h2 className="text-4xl md:text-5xl font-bold mb-4">How It Works</h2>
                    <p className="text-2xl text-gray-400 max-w-3xl mx-auto">
                        Master autonomous racing in three stages.
                    </p>
                </div>

                <div className="flex flex-col md:flex-row items-center md:items-start justify-center gap-8 relative">
                    <div className="flex flex-col items-center text-center max-w-sm relative group">
                        <div className="w-80 h-52 relative rounded-xl overflow-hidden border-2 border-brand-blue/30 mb-5 shadow-2xl shadow-brand-blue/10 group-hover:border-brand-blue transition-colors duration-300">
                            <Image
                                src="/images/car_chassis.png"
                                alt="Build a battery-powered car"
                                fill
                                className="object-cover"
                            />
                        </div>
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300 mb-3">Build</h3>
                        <p className="text-gray-300 text-xl font-medium">
                            Build a battery-powered car with a camera
                        </p>
                    </div>

                    <div className="hidden md:flex items-center justify-center text-brand-blue/50 md:mt-[4.75rem]">
                        <ArrowRight size={56} strokeWidth={1.5} />
                    </div>
                    <div className="md:hidden transform rotate-90 text-brand-blue/50 my-6">
                        <ArrowRight size={56} strokeWidth={1.5} />
                    </div>

                    <div className="flex flex-col items-center text-center max-w-sm relative group">
                        <div className="w-80 h-52 relative rounded-xl overflow-hidden border-2 border-brand-magenta/30 mb-5 shadow-2xl shadow-brand-magenta/10 group-hover:border-brand-magenta transition-colors duration-300">
                            <Image
                                src="/images/code_ros_2.png"
                                alt="Write autonomous racing algorithms in VS Code"
                                fill
                                className="object-cover"
                            />
                        </div>
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300 mb-3">Code</h3>
                        <p className="text-gray-300 text-xl font-medium">
                            Write autonomous racing algorithms
                        </p>
                    </div>

                    <div className="hidden md:flex items-center justify-center text-brand-magenta/50 md:mt-[4.75rem]">
                        <ArrowRight size={56} strokeWidth={1.5} />
                    </div>
                    <div className="md:hidden transform rotate-90 text-brand-magenta/50 my-6">
                        <ArrowRight size={56} strokeWidth={1.5} />
                    </div>

                    <div className="flex flex-col items-center text-center max-w-sm relative group">
                        <div className="w-80 h-52 relative rounded-xl overflow-hidden border-2 border-indigo-500/30 mb-5 shadow-2xl shadow-indigo-500/10 group-hover:border-indigo-500 transition-colors duration-300">
                            <Image
                                src="/images/race_cars.png"
                                alt="Autonomous cars racing on track"
                                fill
                                className="object-cover"
                            />
                        </div>
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300 mb-3">Race</h3>
                        <p className="text-gray-300 text-xl font-medium">
                            Compete in races with teams from across the world
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
