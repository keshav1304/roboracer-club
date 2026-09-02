import Link from "next/link";
import Image from "next/image";
import { Code2, ExternalLink, Eye, Route, SlidersHorizontal, Trophy } from "lucide-react";
import ClubHeroTitle from "@/components/home/ClubHeroTitle";
import HowItWorks from "@/components/home/HowItWorks";

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative min-h-[70vh] flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 z-0">
          <Image
            src="/images/hero_background.png"
            alt="Penn RoboRacer Club"
            fill
            className="object-cover opacity-60"
            priority
          />
          <div className="absolute inset-0 bg-brand-dark/50" />
        </div>

        <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center py-16">
          <ClubHeroTitle />
          <p className="text-xl md:text-2xl text-gray-300 leading-relaxed">
            A club of Penn undergraduates and graduate students exploring
            autonomous, driverless racing. Perception, planning, and control on the track.
          </p>
        </div>
      </section>

      <div className="bg-gradient-to-b from-brand-dark to-brand-[#ec4899]">
        {/* Who We Are */}
        <section className="py-14 border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-10">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">Who We Are</h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                Penn RoboRacer Club is a group of Penn undergraduates and graduate
                students working on autonomous, driverless racing. We do two things:
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">
              <div className="p-7 rounded-3xl bg-white/5 border border-brand-blue/50">
                <div className="w-14 h-14 rounded-xl bg-brand-blue/10 flex items-center justify-center mb-4">
                  <Trophy className="h-8 w-8 text-brand-blue" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Autonomous Racing Competitions</h3>
                <p className="text-lg text-gray-400 leading-relaxed">
                  Autonomous racing competitions. Full stacks for perception,
                  localization, planning, and control. Tested against teams
                  worldwide in simulation and on the track.
                </p>
              </div>

              <div className="p-7 rounded-3xl bg-white/5 border border-brand-magenta/50">
                <div className="w-14 h-14 rounded-xl bg-brand-magenta/10 flex items-center justify-center mb-4">
                  <Code2 className="h-8 w-8 text-brand-magenta" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Open-Source Educational Tools</h3>
                <p className="text-lg text-gray-400 leading-relaxed mb-4">
                  Open-source educational tools for autonomous racing. Hardware,
                  software, and documentation students can learn from and build on.
                </p>
                <p className="text-lg text-gray-300 leading-relaxed">
                  Our current project is{" "}
                  <Link
                    href="/roboracer-mini"
                    className="text-brand-magenta hover:text-brand-magenta-hover font-medium"
                  >
                    RoboRacer-mini
                  </Link>
                  .
                </p>
              </div>
            </div>
          </div>
        </section>

        <HowItWorks />

        {/* Perception, Planning, Control */}
        <section className="py-14 border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-10">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">What We Explore</h2>
              <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                The core problems of autonomous racing, end to end.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-7 rounded-3xl bg-white/5 border border-brand-blue/50">
                <div className="w-14 h-14 rounded-xl bg-brand-blue/10 flex items-center justify-center mb-4">
                  <Eye className="h-8 w-8 text-brand-blue" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Perception</h3>
                <p className="text-lg text-gray-400 leading-relaxed">
                  Cameras, lidar, sensor fusion. Track, opponents, obstacles in real time.
                </p>
              </div>

              <div className="p-7 rounded-3xl bg-white/5 border border-brand-magenta/50">
                <div className="w-14 h-14 rounded-xl bg-brand-magenta/10 flex items-center justify-center mb-4">
                  <Route className="h-8 w-8 text-brand-magenta" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Planning</h3>
                <p className="text-lg text-gray-400 leading-relaxed">
                  Racing lines, overtaking, lap strategy. Perception into competitive trajectories.
                </p>
              </div>

              <div className="p-7 rounded-3xl bg-white/5 border border-brand-blue/50">
                <div className="w-14 h-14 rounded-xl bg-brand-blue/10 flex items-center justify-center mb-4">
                  <SlidersHorizontal className="h-8 w-8 text-brand-blue" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Control</h3>
                <p className="text-lg text-gray-400 leading-relaxed">
                  Steering, throttle, stability at the limit. Plans on physical and simulated vehicles.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* RoboRacer platform */}
        <section className="py-14">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-5">Built on RoboRacer</h2>
            <p className="text-xl text-gray-300 leading-relaxed mb-4">
              Our club builds on the knowledge and ecosystem of the{" "}
              <span className="text-white font-medium">RoboRacer</span> platform,
              the open autonomous racing stack used by universities worldwide for
              research, teaching, and competition.
            </p>
            <p className="text-lg text-gray-400 leading-relaxed mb-6">
              From hardware and ROS workflows to race-day tooling, we leverage
              what the broader RoboRacer community has established as we develop
              our own projects at Penn.
            </p>
            <a
              href="https://roboracer.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-8 py-3 rounded-full bg-brand-blue hover:bg-brand-blue-hover text-white font-semibold transition-colors duration-200"
            >
              Learn more at roboracer.ai
              <ExternalLink size={18} />
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
