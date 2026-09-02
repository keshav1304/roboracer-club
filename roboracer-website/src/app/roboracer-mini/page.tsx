import { Cpu, Trophy, Users, Ruler, DollarSign, Eye, Brain, GraduationCap, Zap, ScanLine, ExternalLink, Globe } from "lucide-react";
import PageHero from "@/components/layout/PageHero";
import InvestigationCard from "@/components/roboracer-mini/InvestigationCard";

const carInvestigations = [
  {
    name: "Waveshare PiRacer Pro",
    summary:
      "Commercial DonkeyCar-ready racer. 4WD, RC380 motor, open-source ecosystem on Raspberry Pi 4.",
    status: "Investigating",
    details: [
      { label: "Platform", value: "Waveshare PiRacer Pro AI Kit" },
      { label: "Compute", value: "Raspberry Pi 4 Model B" },
      { label: "Software", value: "DonkeyCar (Keras/TensorFlow, OpenCV)" },
      { label: "OS", value: "Raspberry Pi OS (Raspbian), Python" },
      { label: "Camera (stock)", value: "5 MP HD, 160° FOV wide-angle" },
      { label: "Display", value: "0.91\" OLED (128×32). IP, memory, power." },
      { label: "Drive", value: "Ackermann steering, 4WD, front/rear differentials" },
      { label: "Suspension", value: "Adjustable oil-filled shocks, 4WD independent" },
      { label: "Motor", value: "RC380 carbon-brushed (~15,000 RPM idle)" },
      { label: "Servo", value: "6 kg·cm torque" },
      { label: "Power", value: "8.4 V, 4× 18650 (2S2P, batteries not included)" },
      { label: "Connectivity", value: "2.4 / 5 GHz WiFi, Bluetooth 5.0" },
      { label: "Wheelbase", value: "~174 mm (kit ~25.5 × 14 × 21.5 cm)" },
      { label: "Notes", value: "Off-the-shelf DonkeyCar stack; useful baseline for teaching and rapid prototyping." },
    ],
    documentationHref: "/roboracer-mini/waveshare",
    documentationDescription:
      "Donkeycar on the Waveshare PiRacer Pro: hardware setup, PWM tuning, line following, lane keeping, and evaluation for RoboRacer-mini.",
    videos: [
      {
        title: "Line following demo",
        youtubeId: "U_DXo-ofhXc",
        url: "https://www.youtube.com/shorts/U_DXo-ofhXc",
      },
      {
        title: "Lane keeping demo",
        youtubeId: "HCT4SYDuZ0k",
        url: "https://www.youtube.com/watch?v=HCT4SYDuZ0k",
      },
    ],
  },
  {
    name: "Lehigh E116",
    summary:
      "E116 from Lehigh University. 1/16-scale Traxxas chassis with Jetson and RealSense D435i.",
    status: "Reference Platform",
    details: [
      { label: "Origin", value: "Lehigh University, E116 platform" },
      { label: "Scale", value: "1/16th scale" },
      { label: "Chassis", value: "Traxxas RC chassis" },
      { label: "Compute", value: "NVIDIA Jetson" },
      { label: "Camera", value: "Intel RealSense D435i" },
      { label: "Notes", value: "Established reference design we are studying for RoboRacer-mini hardware and software choices." },
    ],
    documentationHref: "/roboracer-mini/lehigh-e116",
    documentationDescription:
      "Week-by-week E116 journal: Jetson and Traxxas hardware, ROS 2, RealSense, AprilTags, gap follow, and RoboRacer-mini evaluation.",
    videos: [
      {
        title: "Hardware overview",
        youtubeId: "1DGavq1OEdM",
        url: "https://www.youtube.com/watch?v=1DGavq1OEdM",
      },
      {
        title: "Follow the Gap demo",
        youtubeId: "t0ZZ9GUBdJ0",
        url: "https://www.youtube.com/watch?v=t0ZZ9GUBdJ0",
      },
    ],
  },
  {
    name: "Custom 1/18 RC Truck",
    summary:
      "Our in-house build. Customized 1/18-scale RC truck with Raspberry Pi 5 and Intel RealSense D435i.",
    status: "In Development",
    details: [
      { label: "Scale", value: "1/18th scale RC truck" },
      { label: "Chassis", value: "Customized RC truck platform" },
      { label: "Compute", value: "Raspberry Pi 5" },
      { label: "Camera", value: "Intel RealSense D435i" },
      { label: "Motor Control", value: "Custom ESC (model TBD)" },
      { label: "Software Stack", value: "ROS 2 / custom autonomy stack (TBD)" },
      { label: "Notes", value: "Fully custom integration. Chassis, ESC, and sensor mounting designed by the club." },
    ],
    photoSlots: 1,
    videoSlots: 1,
  },
];

export default function RoboRacerMiniPage() {
  return (
    <div className="flex flex-col">
      <PageHero
        title="RoboRacer-mini"
        subtitle="Open-source platform for high-schoolers and undergraduates to master autonomous racing."
        compact
      />

      <div className="bg-gradient-to-b from-brand-dark to-brand-[#ec4899]">
        {/* Inspired by RoboRacer */}
        <section className="py-14 relative z-20 border-b border-white/10">
          <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 text-center">Inspired by RoboRacer</h2>
            <p className="text-xl text-gray-300 leading-relaxed mb-4">
              RoboRacer-mini takes inspiration from{" "}
              <a
                href="https://roboracer.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-brand-blue hover:text-brand-blue-hover font-medium"
              >
                RoboRacer
              </a>
              , formerly known as F1TENTH. The platform was developed at the
              University of Pennsylvania and has grown into an international
              open-source community for autonomous racing education and research.
            </p>
            <p className="text-lg text-gray-400 leading-relaxed mb-10">
              RoboRacer-mini aims to bring that same hands-on learning to a
              smaller, more affordable scale for high-school and undergraduate
              students.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
              <div className="p-6 rounded-2xl bg-white/5 border border-brand-blue/30 text-center">
                <GraduationCap className="h-8 w-8 text-brand-blue mx-auto mb-3" />
                <p className="text-3xl font-bold text-white mb-1">90+</p>
                <p className="text-sm text-gray-400">Universities worldwide</p>
              </div>
              <div className="p-6 rounded-2xl bg-white/5 border border-brand-magenta/30 text-center">
                <Trophy className="h-8 w-8 text-brand-magenta mx-auto mb-3" />
                <p className="text-3xl font-bold text-white mb-1">24+</p>
                <p className="text-sm text-gray-400">International competitions</p>
              </div>
              <div className="p-6 rounded-2xl bg-white/5 border border-white/10 text-center">
                <Globe className="h-8 w-8 text-brand-blue mx-auto mb-3" />
                <p className="text-lg font-bold text-white mb-1">Open ecosystem</p>
                <p className="text-sm text-gray-400">Hardware, software, curriculum</p>
              </div>
            </div>

            <p className="text-center">
              <a
                href="https://roboracer.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-brand-blue hover:text-brand-blue-hover font-medium transition-colors"
              >
                Explore the RoboRacer platform
                <ExternalLink size={16} />
              </a>
            </p>
          </div>
        </section>

        {/* Evolution Section */}
        <section className="py-14 relative z-20 border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-6">The Evolution</h2>
              <p className="text-2xl text-gray-400 max-w-3xl mx-auto">
                Reimagining autonomous racing to be accessible and powerful.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              <div className="bg-brand-dark rounded-3xl p-8 border-2 border-indigo-500/30 relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent" />
                <div className="absolute top-0 right-0 p-4 opacity-20 text-indigo-500">
                  <ScanLine size={120} />
                </div>
                <div className="relative z-10">
                  <h3 className="text-3xl font-bold text-white mb-1">RoboRacer</h3>
                  <p className="text-sm text-indigo-400 uppercase tracking-widest mb-8">Professional Platform</p>

                  <div className="space-y-6">
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-indigo-400"><Ruler size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Scale</p>
                        <p className="text-xl font-bold text-gray-200">1/10th Scale</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-indigo-400"><DollarSign size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Cost</p>
                        <p className="text-xl font-bold text-gray-200">~$3,500</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-indigo-400"><ScanLine size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Sensors</p>
                        <p className="text-xl font-bold text-gray-200">Lidar-based</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-indigo-400"><Brain size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">AI Stack</p>
                        <p className="text-xl font-bold text-gray-200">Reinforcement Learning</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-indigo-400"><GraduationCap size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Audience</p>
                        <p className="text-xl font-bold text-gray-200">Grad Students</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-brand-dark rounded-3xl p-8 border-2 border-brand-blue relative overflow-hidden shadow-2xl shadow-brand-blue/20 group transform md:-translate-x-4 lg:translate-x-0">
                <div className="absolute inset-0 bg-gradient-to-br from-brand-blue/10 to-transparent" />
                <div className="absolute top-0 right-0 p-4 opacity-20 text-brand-blue">
                  <Zap size={120} />
                </div>

                <div className="relative z-10">
                  <h3 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-blue to-brand-magenta mb-1">RoboRacer-mini</h3>
                  <p className="text-sm text-brand-blue uppercase tracking-widest mb-8">The Future</p>

                  <div className="space-y-6">
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-brand-blue"><Ruler size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Scale</p>
                        <p className="text-2xl font-bold text-white">1/16th Scale</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-brand-blue"><DollarSign size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Cost</p>
                        <p className="text-2xl font-bold text-white text-green-400">~$1,000</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-brand-blue"><Eye size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Sensors</p>
                        <p className="text-2xl font-bold text-white">Vision-based</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-brand-blue"><Brain size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">AI Stack</p>
                        <p className="text-2xl font-bold text-white">LLMs + Gen AI Tool</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-10 flex justify-center text-brand-blue"><Users size={24} /></div>
                      <div className="ml-4 flex-1">
                        <p className="text-sm uppercase tracking-wider text-gray-500">Audience</p>
                        <p className="text-2xl font-bold text-white">High-school & Undergrad</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Chassis Investigation */}
        <section className="py-14 relative z-20 border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-6">Chassis Investigation</h2>
              <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Three platforms we are building on or learning from. Waveshare and
                Lehigh have full investigation docs with videos and photos on-site.
                Expand each card for specs and media.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-6">
              {carInvestigations.map((car) => (
                <InvestigationCard key={car.name} {...car} />
              ))}
            </div>
          </div>
        </section>

        {/* Why RoboRacer-mini? */}
        <section className="py-14 relative z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-6">Why RoboRacer-mini?</h2>
              <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                We bridge the gap between theory and practice, providing a hands-on environment for future roboticists.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="p-10 rounded-3xl bg-white/5 border border-brand-blue/50 transition-colors duration-300">
                <div className="w-14 h-14 rounded-xl bg-brand-blue/10 flex items-center justify-center mb-8">
                  <Cpu className="h-8 w-8 text-brand-blue" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Hands-on Learning</h3>
                <p className="text-lg text-gray-400">
                  Real hardware and sensors. Full stack of autonomous systems: perception, planning, control.
                </p>
              </div>

              <div className="p-10 rounded-3xl bg-white/5 border border-brand-magenta/50 transition-colors duration-300">
                <div className="w-14 h-14 rounded-xl bg-brand-magenta/10 flex items-center justify-center mb-8">
                  <Users className="h-8 w-8 text-brand-magenta" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Community First</h3>
                <p className="text-lg text-gray-400">
                  Students, mentors, and industry experts passionate about robotics and AI.
                </p>
              </div>

              <div className="p-10 rounded-3xl bg-white/5 border border-brand-blue/50 transition-colors duration-300">
                <div className="w-14 h-14 rounded-xl bg-brand-blue/10 flex items-center justify-center mb-8">
                  <Trophy className="h-8 w-8 text-brand-blue" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Competitive Racing</h3>
                <p className="text-lg text-gray-400">
                  Test your algorithms on the track. Compete in tournaments and push your algorithms to the limit.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
