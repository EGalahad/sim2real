import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        {
          type: 'doc',
          id: 'getting-started/overview',
          label: 'Overview',
        },
        {
          type: 'doc',
          id: 'getting-started/root-project',
          label: 'Root Project',
        },
        {
          type: 'doc',
          id: 'getting-started/teleop-x86-64',
          label: 'Teleop Project (x86_64 PC)',
        },
        {
          type: 'doc',
          id: 'getting-started/teleop-onboard-orin',
          label: 'Teleop Project (Onboard Orin)',
        },
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/offline-motion-tracking',
        'tutorials/pico-teleoperation',
        'tutorials/motion-recording',
      ],
    },
    'faq',
    {
      type: 'category',
      label: 'Reference',
      items: [
        {
          type: 'doc',
          id: 'tracking_framework',
          label: 'Tracking Framework',
        },
        {
          type: 'doc',
          id: 'pico_to_g1',
          label: 'Pico To G1',
        },
        {
          type: 'doc',
          id: 'teleop_impl',
          label: 'Teleop Implementation',
        },
        {
          type: 'doc',
          id: 'motion_buffer',
          label: 'Motion Buffer',
        },
        {
          type: 'doc',
          id: 'xrobot_grpc_jetpack5',
          label: 'XRobot gRPC JetPack 5',
        },
        {
          type: 'doc',
          id: 'xrobotoolkit_pc_service_arm64_build',
          label: 'XRoboToolkit PC Service ARM64 Build',
        },
      ],
    },
  ],
};

export default sidebars;
