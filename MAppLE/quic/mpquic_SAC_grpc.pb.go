// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.5.1
// - protoc             v3.20.3
// source: mpquic_SAC.proto

package quic

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	MPQUICService_SelectPath_FullMethodName = "/mpquic_SAC.MPQUICService/SelectPath"
)

// MPQUICServiceClient is the client API for MPQUICService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Define the service
type MPQUICServiceClient interface {
	SelectPath(ctx context.Context, in *AggDelivState, opts ...grpc.CallOption) (*PathInfo, error)
}

type mPQUICServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewMPQUICServiceClient(cc grpc.ClientConnInterface) MPQUICServiceClient {
	return &mPQUICServiceClient{cc}
}

func (c *mPQUICServiceClient) SelectPath(ctx context.Context, in *AggDelivState, opts ...grpc.CallOption) (*PathInfo, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(PathInfo)
	err := c.cc.Invoke(ctx, MPQUICService_SelectPath_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// MPQUICServiceServer is the server API for MPQUICService service.
// All implementations must embed UnimplementedMPQUICServiceServer
// for forward compatibility.
//
// Define the service
type MPQUICServiceServer interface {
	SelectPath(context.Context, *AggDelivState) (*PathInfo, error)
	mustEmbedUnimplementedMPQUICServiceServer()
}

// UnimplementedMPQUICServiceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedMPQUICServiceServer struct{}

func (UnimplementedMPQUICServiceServer) SelectPath(context.Context, *AggDelivState) (*PathInfo, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SelectPath not implemented")
}
func (UnimplementedMPQUICServiceServer) mustEmbedUnimplementedMPQUICServiceServer() {}
func (UnimplementedMPQUICServiceServer) testEmbeddedByValue()                       {}

// UnsafeMPQUICServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to MPQUICServiceServer will
// result in compilation errors.
type UnsafeMPQUICServiceServer interface {
	mustEmbedUnimplementedMPQUICServiceServer()
}

func RegisterMPQUICServiceServer(s grpc.ServiceRegistrar, srv MPQUICServiceServer) {
	// If the following call pancis, it indicates UnimplementedMPQUICServiceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&MPQUICService_ServiceDesc, srv)
}

func _MPQUICService_SelectPath_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AggDelivState)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MPQUICServiceServer).SelectPath(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: MPQUICService_SelectPath_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MPQUICServiceServer).SelectPath(ctx, req.(*AggDelivState))
	}
	return interceptor(ctx, in, info, handler)
}

// MPQUICService_ServiceDesc is the grpc.ServiceDesc for MPQUICService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var MPQUICService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "mpquic_SAC.MPQUICService",
	HandlerType: (*MPQUICServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SelectPath",
			Handler:    _MPQUICService_SelectPath_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "mpquic_SAC.proto",
}